# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

# Importing necessary modules from the fairscale library for model parallelism
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,  # Handles column-wise model parallelism
    ParallelEmbedding,  # Embedding layer that supports model parallelism
    RowParallelLinear,  # Handles row-wise model parallelism
)
from torch import nn

# Device determination logic to utilize available hardware acceleration
# This section checks for the availability of CUDA or MPS (Metal Performance Shaders) backends
# and sets the device accordingly to optimize computation.
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA CUDA is available, use GPU
elif torch.backends.mps.is_available():
    device = "mps"  # Apple MPS is available, use Apple Silicon GPU
else:
    device = "cpu"  # Fallback to CPU if no advanced hardware acceleration is available


# Definition of a data class to store model arguments
# This class uses Python's dataclasses to automatically generate special methods like __init__()
@dataclass
class ModelArgs:
    """Data class for storing model arguments with default values and optional types.

    Attributes:
        dim (int): Dimensionality of the model or layers.
        n_layers (int): Number of layers in the model.
        n_heads (int): Number of attention heads in each layer.
        n_kv_heads (Optional[int]): Number of key/value heads, if different from n_heads.
        vocab_size (int): Size of the vocabulary. Set later by tokenizer.
        multiple_of (int): Ensures certain dimensions are multiples of this value for efficiency.
        ffn_dim_multiplier (Optional[float]): Multiplier for the dimension of feed-forward network.
        norm_eps (float): Epsilon value for normalization layers to avoid division by zero.
        rope_theta (float): Theta value for rotary position embeddings.
        max_batch_size (int): Maximum batch size that can be processed.
        max_seq_len (int): Maximum sequence length the model can handle.
    """

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # To be defined later by tokenizer
    multiple_of: int = (
        256  # Ensures SwiGLU hidden layer size is a multiple of a large power of 2
    )
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    """
    RMS Normalization layer implementing root mean square normalization.

    This layer normalizes the input tensor by its root mean square value, adjusted by a small epsilon for numerical stability.

    Attributes:
        dim (int): The dimensionality of the input tensor expected by the layer.
        eps (float): A small epsilon value to prevent division by zero, defaulting to 1e-6.
        weight (torch.nn.Parameter): A learnable weight parameter initialized to ones, which scales the normalized output.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initializes the RMSNorm layer with specified dimension and epsilon.

        Parameters:
            dim (int): The dimensionality of the input tensor.
            eps (float): A small epsilon value for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Private helper method to compute the RMS normalization of the tensor.

        Parameters:
            x (torch.Tensor): The input tensor to normalize.

        Returns:
            torch.Tensor: The RMS normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RMSNorm layer.

        Parameters:
            x (torch.Tensor): The input tensor to normalize.

        Returns:
            torch.Tensor: The normalized tensor, scaled by the learnable weight.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precomputes the frequency cosine and sine values for rotary embeddings.

    This function calculates the cosine and sine values used in rotary position embeddings, which are used to add positional information to the embeddings.

    Parameters:
        dim (int): The dimensionality of the embedding.
        end (int): The sequence length or the number of positions.
        theta (float): A scaling factor for the frequencies, defaulting to 10000.0.

    Returns:
        torch.Tensor: A tensor of precomputed cosine and sine values for rotary embeddings.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the frequency tensor for broadcasting over the input tensor.

    This function adjusts the shape of the frequency tensor so that it can be broadcasted over the input tensor for element-wise operations.

    Parameters:
        freqs_cis (torch.Tensor): The frequency tensor.
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The reshaped frequency tensor ready for broadcasting.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim, "Tensor x must have at least two dimensions."
    assert freqs_cis.shape == (
        x.shape[1],
        x.shape[-1],
    ), "Mismatch in shapes for broadcasting."
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary position embeddings to query and key tensors.

    This function applies rotary position embeddings to the query and key tensors, which helps in maintaining the relative positional information.

    Parameters:
        xq (torch.Tensor): The query tensor.
        xk (torch.Tensor): The key tensor.
        freqs_cis (torch.Tensor): The precomputed frequency cosine and sine values.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The query and key tensors with applied rotary embeddings.
    """
    if not torch.cuda.is_available():
        xq = xq.to("cpu")
        xk = xk.to("cpu")
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(device), xk_out.type_as(xk).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This function repeats the key-value pairs across the head dimension of the tensor.
    It is particularly useful when the number of key-value heads is less than the number of attention heads,
    necessitating a repetition to match the dimensions for subsequent operations.

    Parameters:
        x (torch.Tensor): The input tensor with dimensions (batch_size, seq_len, num_kv_heads, head_dim).
        n_rep (int): The number of times to repeat the key-value pairs across the head dimension.

    Returns:
        torch.Tensor: The tensor with repeated key-value pairs across the head dimension.
    """
    # Extract the dimensions of the input tensor
    batch_size, seq_len, num_kv_heads, head_dim = x.shape

    # If no repetition is needed, return the original tensor
    if n_rep == 1:
        return x

    # Expand and reshape the tensor to repeat the key-value pairs
    repeated_tensor = (
        x.unsqueeze(3)  # Add an extra dimension for repetition
        .expand(
            batch_size, seq_len, num_kv_heads, n_rep, head_dim
        )  # Expand the new dimension
        .reshape(
            batch_size, seq_len, num_kv_heads * n_rep, head_dim
        )  # Merge the repeated dimensions
    )
    return repeated_tensor


class Attention(nn.Module):
    """
    The Attention class implements the multi-head attention mechanism with model parallelism.
    It supports different numbers of key-value heads and attention heads, handling the distribution
    of model parameters across multiple devices in a parallel computing environment.

    Attributes:
        n_kv_heads (int): Number of key-value heads.
        n_local_heads (int): Number of local attention heads, adjusted for model parallelism.
        n_local_kv_heads (int): Number of local key-value heads, adjusted for model parallelism.
        n_rep (int): Repetition factor for key-value pairs if their number is less than the attention heads.
        head_dim (int): Dimension of each head.
        wq, wk, wv, wo (nn.Module): Linear transformations for query, key, value, and output.
        cache_k, cache_v (torch.Tensor): Cached key and value tensors for efficient memory usage.
    """

    def __init__(self, args: ModelArgs):
        """
        Initializes the Attention module with the specified model arguments.

        Parameters:
            args (ModelArgs): Configuration parameters for the model.
        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # Initialize linear transformations with custom initialization and parallel configuration
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # Initialize caches for keys and values
        self.cache_k = torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim
        ).to(device)
        self.cache_v = torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim
        ).to(device)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        The forward pass for the Attention module, applying multi-head attention to the input tensor.

        Parameters:
            x (torch.Tensor): The input tensor.
            start_pos (int): The starting position in the sequence for caching.
            freqs_cis (torch.Tensor): The frequency cosine and sine values for rotary embeddings.
            mask (Optional[torch.Tensor]): An optional mask tensor to exclude certain positions.

        Returns:
            torch.Tensor: The output tensor after applying attention and linear transformations.
        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape for multi-head attention
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Update caches
        self.cache_k = self.cache_k.to(xq.device)
        self.cache_v = self.cache_v.to(xq.device)
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # Retrieve keys and values from cache
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # Repeat key-value pairs if necessary
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Transpose for batch matrix multiplication
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores += mask

        # Apply softmax and compute the final output
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # Apply the final linear transformation
        return self.wo(output)


class FeedForward(nn.Module):
    """
    A feedforward neural network module that applies a sequence of transformations.

    Attributes:
        w1 (ColumnParallelLinear): First linear transformation with column parallelism.
        w2 (RowParallelLinear): Second linear transformation with row parallelism.
        w3 (ColumnParallelLinear): Third linear transformation with column parallelism.

    Methods:
        forward(x): Applies the feedforward network transformations to the input tensor.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        """
        Initializes the FeedForward module with specified dimensions and multipliers.

        Args:
            dim (int): Dimensionality of the input and output.
            hidden_dim (int): Base dimensionality of the hidden layers.
            multiple_of (int): Ensures hidden dimensions are multiples of this value.
            ffn_dim_multiplier (Optional[float]): Optional multiplier for the hidden dimension.
        """
        super().__init__()
        # Adjust hidden_dim based on the provided ffn_dim_multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        else:
            hidden_dim = int(2 * hidden_dim / 3)
        # Ensure hidden_dim is a multiple of multiple_of
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Initialize linear transformations with specified initialization methods
        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeedForward module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the feedforward transformations.
        """
        x1 = self.w1(x)
        x2 = self.w3(x)
        x1_activated = F.silu(x1)
        x3 = x1_activated * x2
        output = self.w2(x3)
        return output


class TransformerBlock(nn.Module):
    """
    A single block of the transformer model containing attention and feedforward layers.

    Attributes:
        attention (Attention): The attention mechanism of the transformer.
        feed_forward (FeedForward): The feedforward network following the attention.
        attention_norm (RMSNorm): Normalization layer for the attention output.
        ffn_norm (RMSNorm): Normalization layer for the feedforward output.

    Methods:
        forward(x, start_pos, freqs_cis, mask): Processes input through the transformer block.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the TransformerBlock with specified layer ID and model arguments.

        Args:
            layer_id (int): Identifier for the transformer layer.
            args (ModelArgs): Configuration arguments for the model.
        """
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for processing.
            freqs_cis (torch.Tensor): Frequency cosine and sine values for rotary embeddings.
            mask (Optional[torch.Tensor]): Optional mask tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the transformer block.
        """
        normed_x = self.attention_norm(x)
        attention_out = self.attention.forward(normed_x, start_pos, freqs_cis, mask)
        h = x + attention_out
        normed_h = self.ffn_norm(h)
        out = h + self.feed_forward.forward(normed_h)
        return out


class Transformer(nn.Module):
    """
    The complete transformer model consisting of multiple layers.

    Attributes:
        layers (torch.nn.ModuleList): List of transformer blocks.
        tok_embeddings (ParallelEmbedding): Token embeddings for input tokens.
        norm (RMSNorm): Final normalization layer.
        output (ColumnParallelLinear): Output linear transformation.

    Methods:
        forward(tokens, start_pos): Processes input tokens through the transformer model.
    """

    def __init__(self, params: ModelArgs):
        """
        Initializes the Transformer model with specified parameters.

        Args:
            params (ModelArgs): Configuration parameters for the transformer model.
        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size,
            params.dim,
            init_method=lambda x: x,
        )
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(layer_id, params) for layer_id in range(params.n_layers)]
        )
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        """
        Forward pass of the Transformer model.

        Args:
            tokens (torch.Tensor): Input tokens.
            start_pos (int): Starting position for processing.

        Returns:
            torch.Tensor: Output tensor after processing through the transformer model.
        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to("cuda" if device == "cuda" else "cpu")
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=torch.device("cpu")
            )
            mask = mask.to(torch.float32).triu(diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(
                h, start_pos, freqs_cis, (mask.to(device) if mask is not None else mask)
            )
        h = self.norm(h)
        output = self.output(h).float()
        return output


########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from typing import Optional
import types, gc, os, time, re
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
current_path = os.path.dirname(os.path.abspath(__file__))

########################################################################################################

if os.environ.get("RWKV_JIT_ON") != "0":
    os.environ["RWKV_JIT_ON"] = "1"
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
    MyStatic = torch.jit.script
else:
    MyModule = torch.nn.Module

    def __nop(ob):
        return ob

    MyFunction = __nop
    MyStatic = __nop

if os.environ.get("RWKV_CUDA_ON") == "1":
    from torch.utils.cpp_extension import load

    try:
        load(
            name=f"wkv_cuda",
            sources=[
                f"{current_path}/cuda/wrapper.cpp",
                f"{current_path}/cuda/operators.cu",
                f"{current_path}/cuda/gemm_fp16_cublas.cpp",
            ],
            verbose=True,
            extra_ldflags=["cublas.lib" if os.name == "nt" else ""],
            extra_cuda_cflags=[
                "--use_fast_math",
                "-O3",
                "--extra-device-vectorization",
            ],
            is_python_module=False,
        )
        DISABLE_CUBLAS_GEMM = False
    except:
        print(
            "Failed to build cuBLAS matmul, falling back to torch.matmul. Small model with fp16 will overflow."
        )
        load(
            name=f"wkv_cuda",
            sources=[
                f"{current_path}/cuda/wrapper.cpp",
                f"{current_path}/cuda/operators.cu",
            ],
            verbose=True,
            extra_cuda_cflags=[
                "--use_fast_math",
                "-O3",
                "--extra-device-vectorization",
            ],
            extra_cflags=["-DDISABLE_CUBLAS_GEMM"],
            is_python_module=False,
        )
        DISABLE_CUBLAS_GEMM = True

    @MyStatic
    def cuda_wkv(T: int, C: int, w, u, k, v, aa, bb, pp):
        assert 1 * C % min(C, 32) == 0
        assert (
            k.dtype == v.dtype == torch.float16 or k.dtype == v.dtype == torch.float32
        )
        assert w.dtype == u.dtype == aa.dtype == bb.dtype == pp.dtype == torch.float32
        w = w.contiguous()
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        y = torch.empty(
            (T, C),
            device=w.device,
            memory_format=torch.contiguous_format,
            dtype=k.dtype,
        )
        torch.ops.rwkv.wkv_forward(1, T, C, w, u, k, v, y, aa, bb, pp)
        return y, aa, bb, pp

    @MyStatic
    def cuda_mm8_seq(B: int, N: int, M: int, x, w, mx, rx, my, ry):
        assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
        assert x.dtype == torch.float32 or x.dtype == torch.float16
        assert w.dtype == torch.uint8
        assert x.shape == (B, N)
        assert w.shape == (N, M)
        assert rx.shape == mx.shape == (M,)
        assert ry.shape == my.shape == (N, 1)
        y = torch.empty((B, M), device=w.device, dtype=x.dtype)
        torch.ops.rwkv.mm8_seq(B, N, M, x, w, mx, rx, my, ry, y)
        return y

    @MyStatic
    def cuda_mm8_one(N: int, M: int, x, w, mx, rx, my, ry):
        assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
        assert x.dtype == torch.float32 or x.dtype == torch.float16
        assert w.dtype == torch.uint8
        assert x.shape == (N,)
        assert w.shape == (N, M)
        assert rx.shape == mx.shape == (M,)
        assert ry.shape == my.shape == (N, 1)
        y = torch.zeros((M,), device=w.device, dtype=torch.float32)
        torch.ops.rwkv.mm8_one(N, M, x, w, mx, rx, my, ry, y)
        return y.to(dtype=x.dtype)

else:
    os.environ["RWKV_CUDA_ON"] = "0"


@MyStatic
def torch_mm8_seq(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)


@MyStatic
def torch_mm8_one(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)


if os.environ.get("RWKV_CUDA_ON") == "1":

    @MyStatic
    def mm8_seq(x, w, mx, rx, my, ry):
        if w.device.type == "cuda" and x.dtype == torch.float16:
            B, N, M = x.shape[0], w.shape[0], w.shape[1]
            return cuda_mm8_seq(B, N, M, x, w, mx, rx, my, ry)
        else:
            return torch_mm8_seq(x, w, mx, rx, my, ry)

    @MyStatic
    def mm8_one(x, w, mx, rx, my, ry):
        if w.device.type == "cuda":
            N, M = w.shape[0], w.shape[1]
            return cuda_mm8_one(N, M, x, w, mx, rx, my, ry)
        else:
            return torch_mm8_one(x, w, mx, rx, my, ry)

else:

    @MyStatic
    def mm8_seq(x, w, mx, rx, my, ry):
        return torch_mm8_seq(x, w, mx, rx, my, ry)

    @MyStatic
    def mm8_one(x, w, mx, rx, my, ry):
        return torch_mm8_one(x, w, mx, rx, my, ry)


def mm8(
    x: torch.Tensor,
    w: torch.Tensor,
    mx: torch.Tensor,
    rx: torch.Tensor,
    my: torch.Tensor,
    ry: torch.Tensor,
):
    if len(x.shape) == 1:
        return mm8_one(x, w, mx, rx, my, ry)
    return mm8_seq(x, w, mx, rx, my, ry)


def matmul(
    a,
    b,
    mx: Optional[torch.Tensor] = None,
    rx: Optional[torch.Tensor] = None,
    my: Optional[torch.Tensor] = None,
    ry: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = a.dtype
    if b.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        assert a.dtype == b.dtype
        return matmul_float(a, b, output_dtype=output_dtype)
    elif b.dtype == torch.uint8:
        assert mx is not None
        assert rx is not None
        assert my is not None
        assert ry is not None
        return mm8(a, b, mx, rx, my, ry).to(output_dtype)
    else:
        raise ValueError("Unsupported dtype")


if os.environ.get("RWKV_CUDA_ON") == "1" and not DISABLE_CUBLAS_GEMM:

    def matmul_float(a, b, output_dtype: Optional[torch.dtype] = None):
        if output_dtype is None:
            output_dtype = a.dtype
        if a.dtype == b.dtype == torch.float16 and a.device.type == "cuda":
            if len(a.shape) == 1:
                assert len(b.shape) == 2
                c = torch.empty((b.shape[-1],), dtype=output_dtype, device=a.device)
                a = a.unsqueeze(0)
            else:
                assert len(a.shape) == len(b.shape)
                assert len(a.shape) == 2 or len(a.shape) == 3
                # torch.empty((*a.shape[:-1], b.shape[-1])) doesn't work with jit
                if len(a.shape) == 2:
                    c = torch.empty(
                        (a.shape[0], b.shape[-1]), dtype=output_dtype, device=a.device
                    )
                else:
                    c = torch.empty(
                        (a.shape[0], a.shape[1], b.shape[-1]),
                        dtype=output_dtype,
                        device=a.device,
                    )
            torch.ops.rwkv.gemm_fp16_cublas(a, b, c)
            return c
        else:
            return (a @ b).to(output_dtype)

else:

    def matmul_float(a, b, output_dtype: Optional[torch.dtype] = None):
        return (a @ b).to(output_dtype)


if os.environ.get("RWKV_DML_ON") == "1":
    import torch_directml

    print("PyTorch with DirectML Enabled")

########################################################################################################


class RWKV(MyModule):
    def __init__(self, model, strategy, verbose=True, convert_and_save_and_exit=None):
        super().__init__()
        if verbose:
            prxxx = lambda *args, **kwargs: print(*args, **kwargs)
        else:
            prxxx = lambda *args, **kwargs: None

        STRATEGY_REGEX = r"^(?:(?:^|->) *(?:cuda(?::[\d]+)?|cpu|mps|dml) (?:fp(?:16|32)|bf16)(?:i8|i4|i3)?(?: \*[\d]+\+?)? *)+$"
        if not re.match(STRATEGY_REGEX, strategy):
            raise ValueError(
                "Invalid strategy. Please read https://pypi.org/project/rwkv/"
            )

        strategy = ("->".join([x.strip() for x in strategy.split("->")])).replace(
            "->", " -> "
        )
        self.args = types.SimpleNamespace()
        args = self.args
        args.MODEL_NAME = model
        args.strategy_string = strategy

        # Rescale for fp16 mode: set x = x/2 every X layer (to avoid fp16 overflow)
        try:
            self.RESCALE_LAYER = int(
                os.environ["RWKV_RESCALE_LAYER"]
            )  # !!! NOTE: SEEMS YOU SHOULD SET IT TO 999 (disable) FOR RWKV-MUSIC MODELS !!!
        except:
            self.RESCALE_LAYER = 6 if "fp16" in strategy else 0
        prxxx(
            f'RWKV_JIT_ON {os.environ["RWKV_JIT_ON"]} RWKV_CUDA_ON {os.environ["RWKV_CUDA_ON"]} RESCALE_LAYER {self.RESCALE_LAYER}\n'
        )

        args.MODEL_NAME = args.MODEL_NAME.strip()
        if not args.MODEL_NAME.endswith(".pth"):
            args.MODEL_NAME += ".pth"
        prxxx(f"Loading {args.MODEL_NAME} ...")
        with torch.no_grad():
            self.w = torch.load(
                args.MODEL_NAME, map_location="cpu"
            )  # load model to CPU first
            gc.collect()
            w = self.w

            ALREADY_CONVERTED = False
            if "_strategy" in w:
                ALREADY_CONVERTED = True
                assert (
                    convert_and_save_and_exit == None
                )  # you should only convert a raw model
                prxxx(
                    f"Converted model: strategy {w['_strategy']}, version {w['_version']}\n"
                )
                assert (
                    w["_strategy"] == args.strategy_string
                )  # if you are using a new strategy, re-convert the model
                assert (
                    float(w["_version"]) >= 0.7
                )  # sometimes you should re-convert using latest convert_model.py
                assert (
                    w["_rescale_layer"] == self.RESCALE_LAYER
                )  # must use same RESCALE_LAYER to avoid mistakes
                del w["_strategy"]
                del w["_version"]
                del w["_rescale_layer"]

            args.n_embd = w["emb.weight"].shape[1]
            args.n_att = w["blocks.0.att.key.weight"].shape[
                0
            ]  # note: transposed matrix
            args.n_ffn = w["blocks.0.ffn.key.weight"].shape[
                0
            ]  # note: transposed matrix
            args.n_layer = 0
            keys = list(w.keys())
            self.version = 4
            for x in keys:
                layer_id = int(x.split(".")[1]) if ("blocks." in x) else 0
                args.n_layer = max(args.n_layer, layer_id + 1)
                if "ln_x" in x:
                    self.version = max(5, self.version)
                if "gate.weight" in x:
                    self.version = max(5.1, self.version)
                if int(self.version) == 5 and "att.time_decay" in x:
                    args.n_head = w[x].shape[0]
                    if len(w[x].shape) > 1:
                        if w[x].shape[1] > 1:
                            self.version = max(5.2, self.version)
                if "time_maa" in x:
                    self.version = max(6, self.version)
                if int(self.version) == 6 and "time_faaaa" in x:
                    args.n_head = w[x].shape[0]
            prxxx(f"Model detected: v{self.version:.1f}")

            ####################### Compute strategy

            s = [x.strip().split(" ") for x in strategy.split("->")]
            plan = [0] * len(s)
            stream_i = -1
            stream_count = 0
            to_allocate = args.n_layer + 1
            allocated = 0
            free_slots = 0
            for i in range(len(s)):
                si = s[i]
                si1 = si[1]
                if si1.startswith("fp32"):
                    si[1] = [torch.float]
                elif si1.startswith("fp16"):
                    si[1] = [torch.float16]
                elif si1.startswith("bf16"):
                    si[1] = [torch.bfloat16]
                if si1.endswith("i8"):
                    si[1] += [torch.uint8]
                else:
                    si[1] += [si[1][0]]
                if len(si) > 2:
                    ss = si[2]
                    assert ss.startswith("*")
                    if ss.endswith("+"):
                        plan[i] = int(ss[1:-1])
                        stream_i = i
                    else:
                        plan[i] = int(ss[1:])
                    allocated += plan[i]
                    if allocated >= to_allocate:
                        plan[i] += to_allocate - allocated
                        break
                else:
                    free_slots += 1
            if stream_i < 0:
                if free_slots > 0 and to_allocate > allocated:
                    for i in range(len(s)):
                        if plan[i] == 0:
                            plan[i] = (to_allocate - allocated) // free_slots
                            allocated += plan[i]
                            free_slots -= 1
                if to_allocate > allocated:
                    plan[len(s) - 1] += to_allocate - allocated
            else:
                if to_allocate > allocated:
                    stream_count = to_allocate - allocated
                    plan[stream_i] += stream_count
            prxxx(f"Strategy: (total {args.n_layer}+1={args.n_layer+1} layers)")
            for i in range(len(s)):
                ss = s[i]
                if i != stream_i:
                    prxxx(
                        f'* {ss[0]} {str(ss[1]).replace("torch.","")}, store {plan[i]} layers'
                    )
                else:
                    prxxx(
                        f'* {ss[0]} {str(ss[1]).replace("torch.","")}, store {plan[i]-stream_count} layers, stream {stream_count} layers'
                    )
                plan[i] += 0 if i == 0 else plan[i - 1]
            self.strategy = [None] * (args.n_layer + 1)
            strategy = self.strategy
            for n in range(args.n_layer + 1):
                for i in range(len(s)):
                    if n < plan[i]:
                        strategy[n] = types.SimpleNamespace()
                        strategy[n].device = s[i][0]
                        strategy[n].atype = s[i][1][0]
                        strategy[n].wtype = s[i][1][1]
                        strategy[n].stream = False
                        if strategy[n].device == "dml":
                            strategy[n].device = torch_directml.device()
                        if i == stream_i and n >= (plan[i] - stream_count):
                            strategy[n].stream = True
                        break
                prxxx(
                    f"{n}-{strategy[n].device}-{str(strategy[n].atype).replace('torch.','')}-{str(strategy[n].wtype).replace('torch.','')}{'-stream' if strategy[n].stream else ''}",
                    end=" ",
                )
            prxxx()

            ####################### Load weights to self.w

            if not ALREADY_CONVERTED:
                try:  # precompute embedding
                    w["emb.weight"] = F.layer_norm(
                        w["emb.weight"],
                        (args.n_embd,),
                        weight=w["blocks.0.ln0.weight"],
                        bias=w["blocks.0.ln0.bias"],
                    )
                except:
                    w["emb.weight"] = F.layer_norm(
                        w["emb.weight"].float(),
                        (args.n_embd,),
                        weight=w["blocks.0.ln0.weight"].float(),
                        bias=w["blocks.0.ln0.bias"].float(),
                    )
                del w["blocks.0.ln0.weight"]
                del w["blocks.0.ln0.bias"]

            print_need_newline = False

            REAL_TIME_FIRST = False
            for x in list(w.keys()):
                if ".time_faaaa" in x:
                    REAL_TIME_FIRST = True
            if REAL_TIME_FIRST:
                w = {
                    (
                        k.replace(".time_faaaa", ".time_first")
                        if ".time_faaaa" in k
                        else k
                    ): v
                    for k, v in w.items()
                }
                self.w = w

            keys = list(w.keys())
            for x in keys:
                w[x].requires_grad = False
                layer_id = int(x.split(".")[1]) if ("blocks." in x) else 0
                if ("ln_out." in x) or ("head." in x):
                    layer_id = args.n_layer
                dd = strategy[layer_id]
                DEVICE = dd.device
                ATYPE = dd.atype
                WTYPE = dd.wtype

                if not ALREADY_CONVERTED:
                    if self.RESCALE_LAYER > 0:
                        if "att.output.weight" in x:
                            w[x] = w[x] / (2 ** int(layer_id // self.RESCALE_LAYER))
                        if "ffn.value.weight" in x:
                            w[x] = w[x] / (2 ** int(layer_id // self.RESCALE_LAYER))

                    if ".time_" in x:
                        w[x] = w[x].squeeze()
                    if (
                        "key.weight" in x
                        or "value.weight" in x
                        or "receptance.weight" in x
                        or "gate.weight" in x
                        or "output.weight" in x
                        or "head.weight" in x
                    ):
                        w[x] = w[x].t()

                    if ".time_decay" in x and "_w" not in x:  # need fp32 for this
                        if self.version == 4:
                            w[x] = -torch.exp(w[x].float())
                        elif int(self.version) == 5:
                            w[x] = torch.exp(-torch.exp(w[x].float())).reshape(-1, 1, 1)
                            if self.version == 5.2:
                                w[x] = w[x].reshape(args.n_head, -1, 1)
                        elif self.version == 6.0:
                            w[x] = w[x].float().reshape(args.n_head, -1, 1)
                    elif ".time_first" in x:  # need fp32 for this
                        if self.version == 4:
                            w[x] = w[x].float()
                        elif int(self.version) in [5, 6]:
                            if REAL_TIME_FIRST:
                                w[x] = w[x].float().reshape(-1, 1, 1)
                            else:
                                w[x] = torch.exp(w[x].float()).reshape(-1, 1, 1)
                            if self.version in [5.2, 6.0]:
                                w[x] = w[x].reshape(args.n_head, -1, 1)
                    elif ".ln_x" in x:  # need fp32 for group_norm
                        w[x] = w[x].float()
                    else:
                        if (
                            (len(w[x].shape) == 2)
                            and ("emb" not in x)
                            and ("_w1" not in x)
                            and ("_w2" not in x)
                        ):
                            if WTYPE != torch.uint8:
                                w[x] = w[x].to(dtype=WTYPE)
                            else:
                                w[x] = w[x].float()

                                if w[x].shape[0] > w[x].shape[1]:
                                    w[x + "_my"] = torch.amin(w[x], dim=1).unsqueeze(1)
                                    w[x] = w[x] - w[x + "_my"]
                                    w[x + "_mx"] = torch.amin(w[x], dim=0)
                                    w[x] = w[x] - w[x + "_mx"]
                                    w[x + "_rx"] = torch.amax(w[x], dim=0)
                                    w[x] = w[x] / w[x + "_rx"]
                                    w[x + "_ry"] = torch.amax(w[x], dim=1).unsqueeze(1)
                                    w[x] = w[x] / w[x + "_ry"]
                                else:
                                    w[x + "_mx"] = torch.amin(w[x], dim=0)
                                    w[x] = w[x] - w[x + "_mx"]
                                    w[x + "_my"] = torch.amin(w[x], dim=1).unsqueeze(1)
                                    w[x] = w[x] - w[x + "_my"]
                                    w[x + "_rx"] = torch.amax(w[x], dim=0)
                                    w[x] = w[x] / w[x + "_rx"]
                                    w[x + "_ry"] = torch.amax(w[x], dim=1).unsqueeze(1)
                                    w[x] = w[x] / w[x + "_ry"]

                                w[x] = torch.clip(
                                    torch.floor(w[x] * 256), min=0, max=255
                                ).to(dtype=torch.uint8)
                                w[x + "_mx"] = w[x + "_mx"].to(dtype=ATYPE).contiguous()
                                w[x + "_rx"] = (
                                    (w[x + "_rx"] / 16).to(dtype=ATYPE).contiguous()
                                )
                                w[x + "_my"] = w[x + "_my"].to(dtype=ATYPE).contiguous()
                                w[x + "_ry"] = (
                                    (w[x + "_ry"] / 16).to(dtype=ATYPE).contiguous()
                                )
                        else:
                            w[x] = w[x].to(dtype=ATYPE)

                if convert_and_save_and_exit == None:
                    if "emb." in x:
                        w[x] = w[x].contiguous()
                    elif (dd.stream) and (
                        x.endswith("key.weight")
                        or x.endswith("value.weight")
                        or x.endswith("receptance.weight")
                        or x.endswith("output.weight")
                    ):
                        try:
                            w[x] = (
                                w[x].contiguous().pin_memory()
                            )  # if you see "CUDA error: out of memory" here, that's out of CPU RAM, not VRAM. Get more RAM :)
                        except:
                            print(
                                "Note: You are running out of RAM. Get more CPU RAM. Now this will run much slower."
                            )
                    elif DEVICE != "cpu":
                        w[x] = w[x].to(device=DEVICE).contiguous()

                    if (dd.stream) or (DEVICE != "cpu"):
                        try:
                            w[x + "_mx"] = w[x + "_mx"].to(device=DEVICE).contiguous()
                            w[x + "_rx"] = w[x + "_rx"].to(device=DEVICE).contiguous()
                            w[x + "_my"] = w[x + "_my"].to(device=DEVICE).contiguous()
                            w[x + "_ry"] = w[x + "_ry"].to(device=DEVICE).contiguous()
                        except:
                            pass

                if "ffn.value.weight" in x:
                    gc.collect()
                    if "cuda" in args.strategy_string:
                        torch.cuda.empty_cache()

                shape = [i for i in w[x].shape if i != 1]
                if len(shape) > 1:
                    shape = f" {str(shape[0]).rjust(5)} {str(shape[1]).rjust(5)}"
                else:
                    shape = f" {str(shape[0]).rjust(5)}      "
                if layer_id == 0 or layer_id >= args.n_layer - 1:
                    if print_need_newline:
                        prxxx("\n", end="")
                        print_need_newline = False
                    dt = str(w[x].dtype).replace("torch.", "")
                    dt = (
                        dt.replace("float32", "f32")
                        .replace("bfloat16", "bf16")
                        .replace("float16", "f16")
                        .replace("uint8", "i8")
                    )
                    prxxx(
                        x.ljust(32),
                        dt.rjust(4),
                        str(w[x].device).rjust(8),
                        shape,
                        " (pinned)" if w[x].is_pinned() else "",
                    )
                else:
                    print_need_newline = True
                    prxxx(".", end="", flush=True)

            if convert_and_save_and_exit:
                w["_strategy"] = args.strategy_string
                w["_rescale_layer"] = self.RESCALE_LAYER
                w["_version"] = "0.7"
                if not convert_and_save_and_exit.endswith(".pth"):
                    convert_and_save_and_exit += ".pth"
                prxxx(f"Saving to {convert_and_save_and_exit}...")
                torch.save(w, convert_and_save_and_exit)
                prxxx(f"Converted and saved. Now this will exit.")
                exit(0)

            if self.version == 5.2 and os.environ["RWKV_CUDA_ON"] == "1":
                HEAD_SIZE = args.n_att // args.n_head
                rwkv5 = load(
                    name="rwkv5",
                    sources=[
                        f"{current_path}/cuda/rwkv5_op.cpp",
                        f"{current_path}/cuda/rwkv5.cu",
                    ],
                    verbose=True,
                    extra_cuda_cflags=[
                        "-res-usage",
                        "--use_fast_math",
                        "-O3",
                        "-Xptxas -O3" if os.name != "nt" else "",
                        "--extra-device-vectorization",
                        f"-D_N_={HEAD_SIZE}",
                    ],
                )

                class RWKV_5(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, B, T, C, H, state, r, k, v, w, u):
                        with torch.no_grad():
                            assert HEAD_SIZE == C // H
                            ctx.B = B
                            ctx.T = T
                            ctx.C = C
                            ctx.H = H
                            assert state.dtype == torch.float32
                            assert w.dtype == torch.float32
                            assert r.is_contiguous()
                            assert k.is_contiguous()
                            assert v.is_contiguous()
                            assert w.is_contiguous()
                            assert u.is_contiguous()
                            assert state.is_contiguous()

                            y = torch.empty(
                                (B, T, C),
                                device=w.device,
                                dtype=r.dtype,
                                memory_format=torch.contiguous_format,
                            )
                            if r.dtype == torch.bfloat16:
                                rwkv5.forward_bf16(B, T, C, H, state, r, k, v, w, u, y)
                            elif r.dtype == torch.float16:
                                rwkv5.forward_fp16(B, T, C, H, state, r, k, v, w, u, y)
                            elif r.dtype == torch.float32:
                                rwkv5.forward_fp32(B, T, C, H, state, r, k, v, w, u, y)
                            return y, state

                self.RWKV_5 = RWKV_5

            if self.version == 6.0 and os.environ["RWKV_CUDA_ON"] == "1":
                HEAD_SIZE = args.n_att // args.n_head
                rwkv6 = load(
                    name="rwkv6",
                    sources=[
                        f"{current_path}/cuda/rwkv6_op.cpp",
                        f"{current_path}/cuda/rwkv6.cu",
                    ],
                    verbose=True,
                    extra_cuda_cflags=[
                        "-res-usage",
                        "--use_fast_math",
                        "-O3",
                        "-Xptxas -O3",
                        "--extra-device-vectorization",
                        f"-D_N_={HEAD_SIZE}",
                        f"-D_T_={4096}",
                    ],
                )

                class RWKV_6(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, B, T, C, H, state, r, k, v, w, u):
                        with torch.no_grad():
                            assert HEAD_SIZE == C // H
                            ctx.B = B
                            ctx.T = T
                            ctx.C = C
                            ctx.H = H
                            assert state.dtype == torch.float32
                            assert w.dtype == torch.float32
                            assert r.is_contiguous()
                            assert k.is_contiguous()
                            assert v.is_contiguous()
                            assert w.is_contiguous()
                            assert u.is_contiguous()
                            eew = torch.exp(-torch.exp(w.float())).contiguous()

                            y = torch.empty(
                                (B, T, C),
                                device=w.device,
                                dtype=r.dtype,
                                memory_format=torch.contiguous_format,
                            )
                            if r.dtype == torch.bfloat16:
                                rwkv6.forward_bf16(
                                    B, T, C, H, state, r, k, v, eew, u, y
                                )
                            elif r.dtype == torch.float16:
                                rwkv6.forward_fp16(
                                    B, T, C, H, state, r, k, v, eew, u, y
                                )
                            elif r.dtype == torch.float32:
                                rwkv6.forward_fp32(
                                    B, T, C, H, state, r, k, v, eew, u, y
                                )
                            return y, state

                self.RWKV_6 = RWKV_6

            gc.collect()
            if "cuda" in args.strategy_string:
                torch.cuda.empty_cache()

    def RUN_RWKV_5(self, B, T, C, H, state, r, k, v, w, u):
        return self.RWKV_5.apply(B, T, C, H, state, r, k, v, w, u)

    def RUN_RWKV_6(self, B, T, C, H, state, r, k, v, w, u):
        return self.RWKV_6.apply(B, T, C, H, state, r, k, v, w, u)

    ########################################################################################################

    @MyFunction
    def ffn_one(
        self,
        x,
        sx,
        ln_w,
        ln_b,
        k_mix,
        r_mix,
        kw,
        vw,
        rw,
        kmx,
        krx,
        kmy,
        kry,
        vmx,
        vrx,
        vmy,
        vry,
        rmx,
        rrx,
        rmy,
        rry,
    ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        return x + out, xx

    @MyFunction
    def ffn_seq(
        self,
        x,
        sx,
        ln_w,
        ln_b,
        k_mix,
        r_mix,
        kw,
        vw,
        rw,
        kmx,
        krx,
        kmy,
        kry,
        vmx,
        vrx,
        vmy,
        vry,
        rmx,
        rrx,
        rmy,
        rry,
    ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1, :]))
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        return x + out, xx[-1, :]

    @MyFunction
    def ffn_one_v6(
        self,
        x,
        sx,
        ln_w,
        ln_b,
        k_maa,
        r_maa,
        kw,
        vw,
        rw,
        kmx,
        krx,
        kmy,
        kry,
        vmx,
        vrx,
        vmy,
        vry,
        rmx,
        rrx,
        rmy,
        rry,
    ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = sx - xx
        kx = xx + sx * k_maa
        rx = xx + sx * r_maa

        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        return x + out, xx

    @MyFunction
    def ffn_seq_v6(
        self,
        x,
        sx,
        ln_w,
        ln_b,
        k_maa,
        r_maa,
        kw,
        vw,
        rw,
        kmx,
        krx,
        kmy,
        kry,
        vmx,
        vrx,
        vmy,
        vry,
        rmx,
        rrx,
        rmy,
        rry,
    ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1, :]))
        sx = sx - xx
        kx = xx + sx * k_maa
        rx = xx + sx * r_maa

        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        return x + out, xx[-1, :]

    ########################################################################################################

    @MyFunction
    def att_one(
        self,
        x,
        sx,
        aa,
        bb,
        pp,
        ln_w,
        ln_b,
        k_mix,
        v_mix,
        r_mix,
        t_decay,
        t_first,
        kw,
        vw,
        rw,
        ow,
        kmx,
        krx,
        kmy,
        kry,
        vmx,
        vrx,
        vmy,
        vry,
        rmx,
        rrx,
        rmy,
        rry,
        omx,
        orx,
        omy,
        ory,
    ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)

        ww = t_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2)).to(dtype=x.dtype)
        ww = t_decay + pp
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)

        out = matmul(r * wkv, ow, omx, orx, omy, ory)
        return x + out, xx, e1 * aa + e2 * v, e1 * bb + e2, p

    @MyFunction
    def att_seq(
        self,
        x,
        sx,
        aa,
        bb,
        pp,
        ln_w,
        ln_b,
        k_mix,
        v_mix,
        r_mix,
        t_decay,
        t_first,
        kw,
        vw,
        rw,
        ow,
        kmx,
        krx,
        kmy,
        kry,
        vmx,
        vrx,
        vmy,
        vry,
        rmx,
        rrx,
        rmy,
        rry,
        omx,
        orx,
        omy,
        ory,
    ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1, :]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)

        T = x.shape[0]
        for t in range(T):
            kk = k[t]
            vv = v[t]
            ww = t_first + kk
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            sx[t] = ((e1 * aa + e2 * vv) / (e1 * bb + e2)).to(dtype=x.dtype)
            ww = t_decay + pp
            p = torch.maximum(ww, kk)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kk - p)
            aa = e1 * aa + e2 * vv
            bb = e1 * bb + e2
            pp = p
        out = matmul(r * sx, ow, omx, orx, omy, ory)
        return x + out, xx[-1, :], aa, bb, pp

    ########################################################################################################

    @MyFunction
    def att_one_v5(
        self,
        x,
        sx,
        s,
        ln_w,
        ln_b,
        lx_w,
        lx_b,
        k_mix,
        v_mix,
        r_mix,
        t_decay,
        t_first,
        kw,
        vw,
        rw,
        ow,
        kmx,
        krx,
        kmy,
        kry,
        vmx,
        vrx,
        vmy,
        vry,
        rmx,
        rrx,
        rmy,
        rry,
        omx,
        orx,
        omy,
        ory,
    ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        H = t_decay.shape[0]
        N = x.shape[-1] // H

        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)

        a = matmul(k, v)
        out = r @ (t_first * a + s)
        s = a + t_decay * s

        out = out.flatten()
        out = F.group_norm(
            out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps=64e-5
        ).squeeze(0)
        out = out.to(dtype=x.dtype)
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx, s

    @MyFunction
    def att_seq_v5(
        self,
        x,
        sx,
        s,
        ln_w,
        ln_b,
        lx_w,
        lx_b,
        k_mix,
        v_mix,
        r_mix,
        t_decay,
        t_first,
        kw,
        vw,
        rw,
        ow,
        kmx,
        krx,
        kmy,
        kry,
        vmx,
        vrx,
        vmy,
        vry,
        rmx,
        rrx,
        rmy,
        rry,
        omx,
        orx,
        omy,
        ory,
    ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1, :]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        w = t_decay.reshape(-1, 1)
        u = t_first.reshape(-1, 1)
        ws = w.pow(T).reshape(H, 1, 1)
        ind = torch.arange(T - 1, -1, -1, device=w.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, T).pow(ind)
        wk = w.reshape(H, 1, T)
        wb = wk.transpose(-2, -1).flip(1)
        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, T))
        w = torch.tile(w, [T])
        w = w[:, :-T].reshape(-1, T, 2 * T - 1)
        w = w[:, :, T - 1 :].reshape(H, T, T)

        r = (
            matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32)
            .view(T, H, N)
            .transpose(0, 1)
        )
        k = (
            matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
            .view(T, H, N)
            .permute(1, 2, 0)
        )
        v = (
            matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
            .view(T, H, N)
            .transpose(0, 1)
        )

        out = ((r @ k) * w) @ v + (r @ s) * wb
        s = ws * s + (k * wk) @ v

        out = out.transpose(0, 1).contiguous().reshape(T, H * N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps=64e-5)
        out = out.to(dtype=x.dtype)
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx[-1, :], s

    ########################################################################################################

    @MyFunction
    def att_one_v5_1(
        self,
        x,
        sx,
        s,
        ln_w,
        ln_b,
        lx_w,
        lx_b,
        k_mix,
        v_mix,
        r_mix,
        g_mix,
        t_decay,
        t_first,
        kw,
        vw,
        rw,
        gw,
        ow,
        kmx,
        krx,
        kmy,
        kry,
        vmx,
        vrx,
        vmy,
        vry,
        rmx,
        rrx,
        rmy,
        rry,
        gmx,
        grx,
        gmy,
        gry,
        omx,
        orx,
        omy,
        ory,
    ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        H = t_decay.shape[0]
        N = x.shape[-1] // H

        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        a = matmul(k, v)
        out = r @ (t_first * a + s)
        s = a + t_decay * s

        out = out.flatten()
        out = F.group_norm(
            out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps=64e-5
        ).squeeze(0)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx, s

    @MyFunction
    def att_seq_v5_1(
        self,
        x,
        sx,
        s,
        ln_w,
        ln_b,
        lx_w,
        lx_b,
        k_mix,
        v_mix,
        r_mix,
        g_mix,
        t_decay,
        t_first,
        kw,
        vw,
        rw,
        gw,
        ow,
        kmx,
        krx,
        kmy,
        kry,
        vmx,
        vrx,
        vmy,
        vry,
        rmx,
        rrx,
        rmy,
        rry,
        gmx,
        grx,
        gmy,
        gry,
        omx,
        orx,
        omy,
        ory,
    ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1, :]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        w = t_decay.reshape(-1, 1)
        u = t_first.reshape(-1, 1)
        ws = w.pow(T).reshape(H, 1, 1)
        ind = torch.arange(T - 1, -1, -1, device=w.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, T).pow(ind)
        wk = w.reshape(H, 1, T)
        wb = wk.transpose(-2, -1).flip(1)
        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, T))
        w = torch.tile(w, [T])
        w = w[:, :-T].reshape(-1, T, 2 * T - 1)
        w = w[:, :, T - 1 :].reshape(H, T, T)

        r = (
            matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32)
            .view(T, H, N)
            .transpose(0, 1)
        )
        k = (
            matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
            .view(T, H, N)
            .permute(1, 2, 0)
        )
        v = (
            matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
            .view(T, H, N)
            .transpose(0, 1)
        )
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        out = ((r @ k) * w) @ v + (r @ s) * wb
        s = ws * s + (k * wk) @ v

        out = out.transpose(0, 1).contiguous().reshape(T, H * N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps=64e-5)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx[-1, :], s

    ########################################################################################################

    @MyFunction
    def att_seq_v5_2(
        self,
        x,
        sx,
        s,
        ln_w,
        ln_b,
        lx_w,
        lx_b,
        k_mix,
        v_mix,
        r_mix,
        g_mix,
        t_decay,
        t_first,
        kw,
        vw,
        rw,
        gw,
        ow,
        kmx,
        krx,
        kmy,
        kry,
        vmx,
        vrx,
        vmy,
        vry,
        rmx,
        rrx,
        rmy,
        rry,
        gmx,
        grx,
        gmy,
        gry,
        omx,
        orx,
        omy,
        ory,
    ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1, :]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        r = (
            matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32)
            .view(T, H, N)
            .transpose(0, 1)
        )
        k = (
            matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
            .view(T, H, N)
            .permute(1, 2, 0)
        )
        v = (
            matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
            .view(T, H, N)
            .transpose(0, 1)
        )
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        out = torch.empty((T, H, N), dtype=r.dtype, device=r.device)
        for t in range(T):
            rt = r[:, t : t + 1, :]
            kt = k[:, :, t : t + 1]
            vt = v[:, t : t + 1, :]
            at = matmul(kt, vt)
            out[t] = (rt @ (t_first * at + s)).squeeze(1)
            s = at + t_decay * s

        out = out.reshape(T, H * N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps=64e-5)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx[-1, :], s

    ########################################################################################################

    @MyFunction
    def att_one_v6_0(
        self,
        x,
        sx,
        s,
        ln_w,
        ln_b,
        lx_w,
        lx_b,
        x_maa,
        w_maa,
        k_maa,
        v_maa,
        r_maa,
        g_maa,
        tm_w1,
        tm_w2,
        td_w1,
        td_w2,
        t_decay,
        t_first,
        kw,
        vw,
        rw,
        gw,
        ow,
        kmx,
        krx,
        kmy,
        kry,
        vmx,
        vrx,
        vmy,
        vry,
        rmx,
        rrx,
        rmy,
        rry,
        gmx,
        grx,
        gmy,
        gry,
        omx,
        orx,
        omy,
        ory,
    ):
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)

        sx = sx - xx
        xxx = xx + sx * x_maa
        xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)
        xxx = torch.bmm(xxx, tm_w2).view(5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        wx = xx + sx * (w_maa + mw)
        kx = xx + sx * (k_maa + mk)
        vx = xx + sx * (v_maa + mv)
        rx = xx + sx * (r_maa + mr)
        gx = xx + sx * (g_maa + mg)

        H = t_decay.shape[0]
        N = x.shape[-1] // H

        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        w = t_decay + (torch.tanh(wx @ td_w1) @ td_w2).float().view(H, N, 1)
        w = torch.exp(-torch.exp(w.float()))

        a = matmul(k, v)
        out = r @ (t_first * a + s)
        s = a + w * s

        out = out.flatten()
        out = F.group_norm(
            out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps=64e-5
        ).squeeze(0)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx, s

    @MyFunction
    def att_seq_v6_0(
        self,
        x,
        sx,
        s,
        ln_w,
        ln_b,
        lx_w,
        lx_b,
        x_maa,
        w_maa,
        k_maa,
        v_maa,
        r_maa,
        g_maa,
        tm_w1,
        tm_w2,
        td_w1,
        td_w2,
        t_decay,
        t_first,
        kw,
        vw,
        rw,
        gw,
        ow,
        kmx,
        krx,
        kmy,
        kry,
        vmx,
        vrx,
        vmy,
        vry,
        rmx,
        rrx,
        rmy,
        rry,
        gmx,
        grx,
        gmy,
        gry,
        omx,
        orx,
        omy,
        ory,
    ):
        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = torch.cat((sx.unsqueeze(0), xx[:-1, :])) - xx
        xxx = xx + sx * x_maa
        xxx = torch.tanh(xxx @ tm_w1).view(T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, tm_w2).view(5, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        wx = xx + sx * (w_maa + mw)
        kx = xx + sx * (k_maa + mk)
        vx = xx + sx * (v_maa + mv)
        rx = xx + sx * (r_maa + mr)
        gx = xx + sx * (g_maa + mg)

        r = (
            matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32)
            .view(T, H, N)
            .transpose(0, 1)
        )
        k = (
            matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
            .view(T, H, N)
            .permute(1, 2, 0)
        )
        v = (
            matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
            .view(T, H, N)
            .transpose(0, 1)
        )
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        w = t_decay.view(1, H, N, 1) + (torch.tanh(wx @ td_w1) @ td_w2).float().view(
            T, H, N, 1
        )
        w = torch.exp(-torch.exp(w.float()))
        out = torch.empty((T, H, N), dtype=r.dtype, device=r.device)
        for t in range(T):
            rt = r[:, t : t + 1, :]
            kt = k[:, :, t : t + 1]
            vt = v[:, t : t + 1, :]
            at = matmul(kt, vt)
            out[t] = (rt @ (t_first * at + s)).squeeze(1)
            s = at + w[t] * s

        out = out.reshape(T, H * N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps=64e-5)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx[-1, :], s

    ########################################################################################################

    if os.environ["RWKV_CUDA_ON"] == "1":

        @MyFunction
        def cuda_att_seq(
            self,
            x,
            sx,
            aa,
            bb,
            pp,
            ln_w,
            ln_b,
            k_mix,
            v_mix,
            r_mix,
            t_decay,
            t_first,
            kw,
            vw,
            rw,
            ow,
            kmx,
            krx,
            kmy,
            kry,
            vmx,
            vrx,
            vmy,
            vry,
            rmx,
            rrx,
            rmy,
            rry,
            omx,
            orx,
            omy,
            ory,
        ):
            T, C = x.shape
            xx = F.layer_norm(x, (C,), weight=ln_w, bias=ln_b)
            sx = torch.cat((sx.unsqueeze(0), xx[:-1, :]))
            kx = xx * k_mix + sx * (1 - k_mix)
            vx = xx * v_mix + sx * (1 - v_mix)
            rx = xx * r_mix + sx * (1 - r_mix)

            r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
            k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
            v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
            y, aa, bb, pp = cuda_wkv(T, C, t_decay, t_first, k, v, aa, bb, pp)

            out = matmul(r * y.to(x.dtype), ow, omx, orx, omy, ory)
            return x + out, xx[-1, :], aa, bb, pp

        @MyFunction
        def v5_2_before(
            self,
            x,
            sx,
            s,
            ln_w,
            ln_b,
            lx_w,
            lx_b,
            k_mix,
            v_mix,
            r_mix,
            g_mix,
            t_decay,
            t_first,
            kw,
            vw,
            rw,
            gw,
            ow,
            kmx,
            krx,
            kmy,
            kry,
            vmx,
            vrx,
            vmy,
            vry,
            rmx,
            rrx,
            rmy,
            rry,
            gmx,
            grx,
            gmy,
            gry,
            omx,
            orx,
            omy,
            ory,
        ):
            xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
            sx = torch.cat((sx.unsqueeze(0), xx[:-1, :]))
            kx = xx * k_mix + sx * (1 - k_mix)
            vx = xx * v_mix + sx * (1 - v_mix)
            rx = xx * r_mix + sx * (1 - r_mix)
            gx = xx * g_mix + sx * (1 - g_mix)

            r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32)
            k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
            v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
            g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

            return r, k, v, g, xx[-1, :], s.transpose(-1, -2).contiguous()

        @MyFunction
        def v5_2_after(
            self, t_decay, out, s, x, xxx, g, lx_w, lx_b, ow, omx, orx, omy, ory
        ):
            H = t_decay.shape[0]
            N = x.shape[-1] // H
            T = x.shape[0]

            s = s.transpose(-1, -2)
            out = out.reshape(T, H * N)
            out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps=64e-5)
            out = out.to(dtype=x.dtype) * g
            out = matmul(out, ow, omx, orx, omy, ory)

            return x + out, xxx, s

        def cuda_att_seq_v5_2(
            self,
            x,
            sx,
            s,
            ln_w,
            ln_b,
            lx_w,
            lx_b,
            k_mix,
            v_mix,
            r_mix,
            g_mix,
            t_decay,
            t_first,
            kw,
            vw,
            rw,
            gw,
            ow,
            kmx,
            krx,
            kmy,
            kry,
            vmx,
            vrx,
            vmy,
            vry,
            rmx,
            rrx,
            rmy,
            rry,
            gmx,
            grx,
            gmy,
            gry,
            omx,
            orx,
            omy,
            ory,
        ):
            H = t_decay.shape[0]
            N = x.shape[-1] // H
            T = x.shape[0]

            r, k, v, g, xxx, ss = self.v5_2_before(
                x,
                sx,
                s,
                ln_w,
                ln_b,
                lx_w,
                lx_b,
                k_mix,
                v_mix,
                r_mix,
                g_mix,
                t_decay,
                t_first,
                kw,
                vw,
                rw,
                gw,
                ow,
                kmx,
                krx,
                kmy,
                kry,
                vmx,
                vrx,
                vmy,
                vry,
                rmx,
                rrx,
                rmy,
                rry,
                gmx,
                grx,
                gmy,
                gry,
                omx,
                orx,
                omy,
                ory,
            )

            out, s = self.RUN_RWKV_5(
                1, T, self.args.n_att, H, ss, r, k, v, w=t_decay, u=t_first
            )

            return self.v5_2_after(
                t_decay, out, s, x, xxx, g, lx_w, lx_b, ow, omx, orx, omy, ory
            )

        @MyFunction
        def v6_0_before(
            self,
            x,
            sx,
            s,
            ln_w,
            ln_b,
            lx_w,
            lx_b,
            x_maa,
            w_maa,
            k_maa,
            v_maa,
            r_maa,
            g_maa,
            tm_w1,
            tm_w2,
            td_w1,
            td_w2,
            t_decay,
            t_first,
            kw,
            vw,
            rw,
            gw,
            ow,
            kmx,
            krx,
            kmy,
            kry,
            vmx,
            vrx,
            vmy,
            vry,
            rmx,
            rrx,
            rmy,
            rry,
            gmx,
            grx,
            gmy,
            gry,
            omx,
            orx,
            omy,
            ory,
        ):
            H = t_decay.shape[0]
            N = x.shape[-1] // H
            T = x.shape[0]

            xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
            sx = torch.cat((sx.unsqueeze(0), xx[:-1, :])) - xx
            xxx = xx + sx * x_maa
            xxx = torch.tanh(xxx @ tm_w1).view(T, 5, -1).transpose(0, 1)
            xxx = torch.bmm(xxx, tm_w2).view(5, T, -1)
            mw, mk, mv, mr, mg = xxx.unbind(dim=0)

            wx = xx + sx * (w_maa + mw)
            kx = xx + sx * (k_maa + mk)
            vx = xx + sx * (v_maa + mv)
            rx = xx + sx * (r_maa + mr)
            gx = xx + sx * (g_maa + mg)

            r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32)
            k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
            v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
            g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

            w = t_decay.view(1, H, N, 1) + (
                torch.tanh(wx @ td_w1) @ td_w2
            ).float().view(T, H, N, 1)

            return r, k, v, g, w, xx[-1, :], s.transpose(-1, -2).contiguous()

        def cuda_att_seq_v6_0(
            self,
            x,
            sx,
            s,
            ln_w,
            ln_b,
            lx_w,
            lx_b,
            x_maa,
            w_maa,
            k_maa,
            v_maa,
            r_maa,
            g_maa,
            tm_w1,
            tm_w2,
            td_w1,
            td_w2,
            t_decay,
            t_first,
            kw,
            vw,
            rw,
            gw,
            ow,
            kmx,
            krx,
            kmy,
            kry,
            vmx,
            vrx,
            vmy,
            vry,
            rmx,
            rrx,
            rmy,
            rry,
            gmx,
            grx,
            gmy,
            gry,
            omx,
            orx,
            omy,
            ory,
        ):
            H = t_decay.shape[0]
            N = x.shape[-1] // H
            T = x.shape[0]

            r, k, v, g, w, xxx, ss = self.v6_0_before(
                x,
                sx,
                s,
                ln_w,
                ln_b,
                lx_w,
                lx_b,
                x_maa,
                w_maa,
                k_maa,
                v_maa,
                r_maa,
                g_maa,
                tm_w1,
                tm_w2,
                td_w1,
                td_w2,
                t_decay,
                t_first,
                kw,
                vw,
                rw,
                gw,
                ow,
                kmx,
                krx,
                kmy,
                kry,
                vmx,
                vrx,
                vmy,
                vry,
                rmx,
                rrx,
                rmy,
                rry,
                gmx,
                grx,
                gmy,
                gry,
                omx,
                orx,
                omy,
                ory,
            )

            out, s = self.RUN_RWKV_6(
                1, T, self.args.n_att, H, ss, r, k, v, w=w, u=t_first
            )
            return self.v5_2_after(
                t_decay, out, s, x, xxx, g, lx_w, lx_b, ow, omx, orx, omy, ory
            )

    ########################################################################################################

    def forward(self, tokens, state, full_output=False):
        with torch.no_grad():
            w = self.w
            args = self.args

            if state == None:
                if self.version == 4:
                    state = [None] * args.n_layer * 5
                    for i in range(
                        args.n_layer
                    ):  # state: 0=att_xx 1=att_aa 2=att_bb 3=att_pp 4=ffn_xx
                        dd = self.strategy[i]
                        dev = dd.device
                        atype = dd.atype
                        state[i * 5 + 0] = torch.zeros(
                            args.n_embd, dtype=atype, requires_grad=False, device=dev
                        ).contiguous()
                        state[i * 5 + 1] = torch.zeros(
                            args.n_att,
                            dtype=torch.float,
                            requires_grad=False,
                            device=dev,
                        ).contiguous()
                        state[i * 5 + 2] = torch.zeros(
                            args.n_att,
                            dtype=torch.float,
                            requires_grad=False,
                            device=dev,
                        ).contiguous()
                        state[i * 5 + 3] = (
                            torch.zeros(
                                args.n_att,
                                dtype=torch.float,
                                requires_grad=False,
                                device=dev,
                            ).contiguous()
                            - 1e30
                        )
                        state[i * 5 + 4] = torch.zeros(
                            args.n_embd, dtype=atype, requires_grad=False, device=dev
                        ).contiguous()
                elif int(self.version) in [5, 6]:
                    state = [None] * args.n_layer * 3
                    for i in range(args.n_layer):  # state: 0=att_xx 1=att_kv 2=ffn_xx
                        dd = self.strategy[i]
                        dev = dd.device
                        atype = dd.atype
                        state[i * 3 + 0] = torch.zeros(
                            args.n_embd, dtype=atype, requires_grad=False, device=dev
                        ).contiguous()
                        state[i * 3 + 1] = torch.zeros(
                            (
                                args.n_head,
                                args.n_att // args.n_head,
                                args.n_att // args.n_head,
                            ),
                            dtype=torch.float,
                            requires_grad=False,
                            device=dev,
                        ).contiguous()
                        state[i * 3 + 2] = torch.zeros(
                            args.n_embd, dtype=atype, requires_grad=False, device=dev
                        ).contiguous()

            seq_mode = len(tokens) > 1

            x = w["emb.weight"][tokens if seq_mode else tokens[0]]

            for i in range(args.n_layer):
                bbb = f"blocks.{i}."
                att = f"blocks.{i}.att."
                ffn = f"blocks.{i}.ffn."
                dd = self.strategy[i]
                dev = dd.device
                atype = dd.atype
                wtype = dd.wtype
                if seq_mode:
                    cuda_applicable = os.environ[
                        "RWKV_CUDA_ON"
                    ] == "1" and "cuda" in str(dev)
                    if cuda_applicable:
                        ATT = self.cuda_att_seq
                    else:
                        ATT = self.att_seq
                    if self.version == 5:
                        ATT = self.att_seq_v5
                    elif self.version == 5.1:
                        ATT = self.att_seq_v5_1
                    elif self.version == 5.2:
                        ATT = self.att_seq_v5_2
                        if cuda_applicable:
                            ATT = self.cuda_att_seq_v5_2
                    elif self.version == 6.0:
                        ATT = self.att_seq_v6_0
                        if cuda_applicable:
                            ATT = self.cuda_att_seq_v6_0
                    FFN = self.ffn_seq
                    if self.version >= 6.0:
                        FFN = self.ffn_seq_v6
                else:
                    ATT = self.att_one
                    if self.version == 5:
                        ATT = self.att_one_v5
                    elif self.version == 5.1:
                        ATT = self.att_one_v5_1
                    elif self.version == 5.2:
                        ATT = self.att_one_v5_1  # same as v5.1
                    elif self.version == 6.0:
                        ATT = self.att_one_v6_0
                    FFN = self.ffn_one
                    if self.version >= 6.0:
                        FFN = self.ffn_one_v6

                x = x.to(dtype=atype, device=dev)

                kw = w[f"{att}key.weight"]
                vw = w[f"{att}value.weight"]
                rw = w[f"{att}receptance.weight"]
                ow = w[f"{att}output.weight"]
                if dd.stream:
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)
                    rw = rw.to(device=dev, non_blocking=True)
                    ow = ow.to(device=dev, non_blocking=True)
                kmx = w[f"{att}key.weight_mx"] if wtype == torch.uint8 else x
                krx = w[f"{att}key.weight_rx"] if wtype == torch.uint8 else x
                kmy = w[f"{att}key.weight_my"] if wtype == torch.uint8 else x
                kry = w[f"{att}key.weight_ry"] if wtype == torch.uint8 else x
                vmx = w[f"{att}value.weight_mx"] if wtype == torch.uint8 else x
                vrx = w[f"{att}value.weight_rx"] if wtype == torch.uint8 else x
                vmy = w[f"{att}value.weight_my"] if wtype == torch.uint8 else x
                vry = w[f"{att}value.weight_ry"] if wtype == torch.uint8 else x
                rmx = w[f"{att}receptance.weight_mx"] if wtype == torch.uint8 else x
                rrx = w[f"{att}receptance.weight_rx"] if wtype == torch.uint8 else x
                rmy = w[f"{att}receptance.weight_my"] if wtype == torch.uint8 else x
                rry = w[f"{att}receptance.weight_ry"] if wtype == torch.uint8 else x
                omx = w[f"{att}output.weight_mx"] if wtype == torch.uint8 else x
                orx = w[f"{att}output.weight_rx"] if wtype == torch.uint8 else x
                omy = w[f"{att}output.weight_my"] if wtype == torch.uint8 else x
                ory = w[f"{att}output.weight_ry"] if wtype == torch.uint8 else x
                if self.version in [5.1, 5.2, 6.0]:
                    gw = w[f"{att}gate.weight"]
                    if dd.stream:
                        gw = gw.to(device=dev, non_blocking=True)
                    gmx = w[f"{att}gate.weight_mx"] if wtype == torch.uint8 else x
                    grx = w[f"{att}gate.weight_rx"] if wtype == torch.uint8 else x
                    gmy = w[f"{att}gate.weight_my"] if wtype == torch.uint8 else x
                    gry = w[f"{att}gate.weight_ry"] if wtype == torch.uint8 else x
                if self.version == 4:
                    (
                        x,
                        state[i * 5 + 0],
                        state[i * 5 + 1],
                        state[i * 5 + 2],
                        state[i * 5 + 3],
                    ) = ATT(
                        x,
                        state[i * 5 + 0],
                        state[i * 5 + 1],
                        state[i * 5 + 2],
                        state[i * 5 + 3],
                        w[f"{bbb}ln1.weight"],
                        w[f"{bbb}ln1.bias"],
                        w[f"{att}time_mix_k"],
                        w[f"{att}time_mix_v"],
                        w[f"{att}time_mix_r"],
                        w[f"{att}time_decay"],
                        w[f"{att}time_first"],
                        kw,
                        vw,
                        rw,
                        ow,
                        kmx,
                        krx,
                        kmy,
                        kry,
                        vmx,
                        vrx,
                        vmy,
                        vry,
                        rmx,
                        rrx,
                        rmy,
                        rry,
                        omx,
                        orx,
                        omy,
                        ory,
                    )
                elif self.version == 5:
                    x, state[i * 3 + 0], state[i * 3 + 1] = ATT(
                        x,
                        state[i * 3 + 0],
                        state[i * 3 + 1],
                        w[f"{bbb}ln1.weight"],
                        w[f"{bbb}ln1.bias"],
                        w[f"{att}ln_x.weight"],
                        w[f"{att}ln_x.bias"],
                        w[f"{att}time_mix_k"],
                        w[f"{att}time_mix_v"],
                        w[f"{att}time_mix_r"],
                        w[f"{att}time_decay"],
                        w[f"{att}time_first"],
                        kw,
                        vw,
                        rw,
                        ow,
                        kmx,
                        krx,
                        kmy,
                        kry,
                        vmx,
                        vrx,
                        vmy,
                        vry,
                        rmx,
                        rrx,
                        rmy,
                        rry,
                        omx,
                        orx,
                        omy,
                        ory,
                    )
                elif self.version in [5.1, 5.2]:
                    x, state[i * 3 + 0], state[i * 3 + 1] = ATT(
                        x,
                        state[i * 3 + 0],
                        state[i * 3 + 1],
                        w[f"{bbb}ln1.weight"],
                        w[f"{bbb}ln1.bias"],
                        w[f"{att}ln_x.weight"],
                        w[f"{att}ln_x.bias"],
                        w[f"{att}time_mix_k"],
                        w[f"{att}time_mix_v"],
                        w[f"{att}time_mix_r"],
                        w[f"{att}time_mix_g"],
                        w[f"{att}time_decay"],
                        w[f"{att}time_first"],
                        kw,
                        vw,
                        rw,
                        gw,
                        ow,
                        kmx,
                        krx,
                        kmy,
                        kry,
                        vmx,
                        vrx,
                        vmy,
                        vry,
                        rmx,
                        rrx,
                        rmy,
                        rry,
                        gmx,
                        grx,
                        gmy,
                        gry,
                        omx,
                        orx,
                        omy,
                        ory,
                    )
                elif self.version == 6.0:
                    x, state[i * 3 + 0], state[i * 3 + 1] = ATT(
                        x,
                        state[i * 3 + 0],
                        state[i * 3 + 1],
                        w[f"{bbb}ln1.weight"],
                        w[f"{bbb}ln1.bias"],
                        w[f"{att}ln_x.weight"],
                        w[f"{att}ln_x.bias"],
                        w[f"{att}time_maa_x"],
                        w[f"{att}time_maa_w"],
                        w[f"{att}time_maa_k"],
                        w[f"{att}time_maa_v"],
                        w[f"{att}time_maa_r"],
                        w[f"{att}time_maa_g"],
                        w[f"{att}time_maa_w1"],
                        w[f"{att}time_maa_w2"],
                        w[f"{att}time_decay_w1"],
                        w[f"{att}time_decay_w2"],
                        w[f"{att}time_decay"],
                        w[f"{att}time_first"],
                        kw,
                        vw,
                        rw,
                        gw,
                        ow,
                        kmx,
                        krx,
                        kmy,
                        kry,
                        vmx,
                        vrx,
                        vmy,
                        vry,
                        rmx,
                        rrx,
                        rmy,
                        rry,
                        gmx,
                        grx,
                        gmy,
                        gry,
                        omx,
                        orx,
                        omy,
                        ory,
                    )
                if dd.stream:
                    del kw, vw, rw, ow
                    if self.version in [5.1, 5.2, 6.0]:
                        del gw

                kw = w[f"{ffn}key.weight"]
                vw = w[f"{ffn}value.weight"]
                rw = w[f"{ffn}receptance.weight"]
                if dd.stream:
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)
                    rw = rw.to(device=dev, non_blocking=True)
                kmx = w[f"{ffn}key.weight_mx"] if wtype == torch.uint8 else x
                krx = w[f"{ffn}key.weight_rx"] if wtype == torch.uint8 else x
                kmy = w[f"{ffn}key.weight_my"] if wtype == torch.uint8 else x
                kry = w[f"{ffn}key.weight_ry"] if wtype == torch.uint8 else x
                vmx = w[f"{ffn}value.weight_mx"] if wtype == torch.uint8 else x
                vrx = w[f"{ffn}value.weight_rx"] if wtype == torch.uint8 else x
                vmy = w[f"{ffn}value.weight_my"] if wtype == torch.uint8 else x
                vry = w[f"{ffn}value.weight_ry"] if wtype == torch.uint8 else x
                rmx = w[f"{ffn}receptance.weight_mx"] if wtype == torch.uint8 else x
                rrx = w[f"{ffn}receptance.weight_rx"] if wtype == torch.uint8 else x
                rmy = w[f"{ffn}receptance.weight_my"] if wtype == torch.uint8 else x
                rry = w[f"{ffn}receptance.weight_ry"] if wtype == torch.uint8 else x
                if self.version == 4:
                    offset = i * 5 + 4
                elif int(self.version) in [5, 6]:
                    offset = i * 3 + 2
                if self.version < 6.0:
                    x, state[offset] = FFN(
                        x,
                        state[offset],
                        w[f"{bbb}ln2.weight"],
                        w[f"{bbb}ln2.bias"],
                        w[f"{ffn}time_mix_k"],
                        w[f"{ffn}time_mix_r"],
                        kw,
                        vw,
                        rw,
                        kmx,
                        krx,
                        kmy,
                        kry,
                        vmx,
                        vrx,
                        vmy,
                        vry,
                        rmx,
                        rrx,
                        rmy,
                        rry,
                    )
                else:
                    x, state[offset] = FFN(
                        x,
                        state[offset],
                        w[f"{bbb}ln2.weight"],
                        w[f"{bbb}ln2.bias"],
                        w[f"{ffn}time_maa_k"],
                        w[f"{ffn}time_maa_r"],
                        kw,
                        vw,
                        rw,
                        kmx,
                        krx,
                        kmy,
                        kry,
                        vmx,
                        vrx,
                        vmy,
                        vry,
                        rmx,
                        rrx,
                        rmy,
                        rry,
                    )
                if dd.stream:
                    del kw, vw, rw

                if self.RESCALE_LAYER > 0:
                    if (i + 1) % self.RESCALE_LAYER == 0:
                        x = x / 2

            dd = self.strategy[args.n_layer]
            x = x[-1, :] if (seq_mode and (not full_output)) else x
            x = x.to(dtype=dd.atype, device=dd.device)

            x = F.layer_norm(
                x, (args.n_embd,), weight=w["ln_out.weight"], bias=w["ln_out.bias"]
            )
            if w["head.weight"].dtype != torch.uint8:
                x = x @ w["head.weight"]
            else:
                if seq_mode and full_output:
                    x = mm8_seq(
                        x,
                        w["head.weight"],
                        w["head.weight_mx"],
                        w["head.weight_rx"],
                        w["head.weight_my"],
                        w["head.weight_ry"],
                    )
                else:
                    x = mm8_one(
                        x,
                        w["head.weight"],
                        w["head.weight_mx"],
                        w["head.weight_rx"],
                        w["head.weight_my"],
                        w["head.weight_ry"],
                    )

            return x.float(), state
