from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
from torchaudio.models import Emformer
class _EmformerEncoder(torch.nn.Module, _Transcriber):
    """Emformer-based recurrent neural network transducer (RNN-T) encoder (transcription network).

    Args:
        input_dim (int): feature dimension of each input sequence element.
        output_dim (int): feature dimension of each output sequence element.
        segment_length (int): length of input segment expressed as number of frames.
        right_context_length (int): length of right context expressed as number of frames.
        time_reduction_input_dim (int): dimension to scale each element in input sequences to
            prior to applying time reduction block.
        time_reduction_stride (int): factor by which to reduce length of input sequence.
        transformer_num_heads (int): number of attention heads in each Emformer layer.
        transformer_ffn_dim (int): hidden layer dimension of each Emformer layer's feedforward network.
        transformer_num_layers (int): number of Emformer layers to instantiate.
        transformer_left_context_length (int): length of left context.
        transformer_dropout (float, optional): transformer dropout probability. (Default: 0.0)
        transformer_activation (str, optional): activation function to use in each Emformer layer's
            feedforward network. Must be one of ("relu", "gelu", "silu"). (Default: "relu")
        transformer_max_memory_size (int, optional): maximum number of memory elements to use. (Default: 0)
        transformer_weight_init_scale_strategy (str, optional): per-layer weight initialization scaling
            strategy. Must be one of ("depthwise", "constant", ``None``). (Default: "depthwise")
        transformer_tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
    """

    def __init__(self, *, input_dim: int, output_dim: int, segment_length: int, right_context_length: int, time_reduction_input_dim: int, time_reduction_stride: int, transformer_num_heads: int, transformer_ffn_dim: int, transformer_num_layers: int, transformer_left_context_length: int, transformer_dropout: float=0.0, transformer_activation: str='relu', transformer_max_memory_size: int=0, transformer_weight_init_scale_strategy: str='depthwise', transformer_tanh_on_mem: bool=False) -> None:
        super().__init__()
        self.input_linear = torch.nn.Linear(input_dim, time_reduction_input_dim, bias=False)
        self.time_reduction = _TimeReduction(time_reduction_stride)
        transformer_input_dim = time_reduction_input_dim * time_reduction_stride
        self.transformer = Emformer(transformer_input_dim, transformer_num_heads, transformer_ffn_dim, transformer_num_layers, segment_length // time_reduction_stride, dropout=transformer_dropout, activation=transformer_activation, left_context_length=transformer_left_context_length, right_context_length=right_context_length // time_reduction_stride, max_memory_size=transformer_max_memory_size, weight_init_scale_strategy=transformer_weight_init_scale_strategy, tanh_on_mem=transformer_tanh_on_mem)
        self.output_linear = torch.nn.Linear(transformer_input_dim, output_dim)
        self.layer_norm = torch.nn.LayerNorm(output_dim)

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        B: batch size;
        T: maximum input sequence length in batch;
        D: feature dimension of each input sequence frame (input_dim).

        Args:
            input (torch.Tensor): input frame sequences right-padded with right context, with
                shape `(B, T + right context length, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output input lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output frame sequences.
        """
        input_linear_out = self.input_linear(input)
        time_reduction_out, time_reduction_lengths = self.time_reduction(input_linear_out, lengths)
        transformer_out, transformer_lengths = self.transformer(time_reduction_out, time_reduction_lengths)
        output_linear_out = self.output_linear(transformer_out)
        layer_norm_out = self.layer_norm(output_linear_out)
        return (layer_norm_out, transformer_lengths)

    @torch.jit.export
    def infer(self, input: torch.Tensor, lengths: torch.Tensor, states: Optional[List[List[torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """Forward pass for inference.

        B: batch size;
        T: maximum input sequence segment length in batch;
        D: feature dimension of each input sequence frame (input_dim).

        Args:
            input (torch.Tensor): input frame sequence segments right-padded with right context, with
                shape `(B, T + right context length, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``infer``.

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output input lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation
                    of ``infer``.
        """
        input_linear_out = self.input_linear(input)
        time_reduction_out, time_reduction_lengths = self.time_reduction(input_linear_out, lengths)
        transformer_out, transformer_lengths, transformer_states = self.transformer.infer(time_reduction_out, time_reduction_lengths, states)
        output_linear_out = self.output_linear(transformer_out)
        layer_norm_out = self.layer_norm(output_linear_out)
        return (layer_norm_out, transformer_lengths, transformer_states)