import math
from typing import List, Optional, Tuple
import torch
from torchaudio.models.emformer import _EmformerAttention, _EmformerImpl, _get_weight_init_gains
class ConvEmformer(_EmformerImpl):
    """Implements the convolution-augmented streaming transformer architecture introduced in
    *Streaming Transformer Transducer based Speech Recognition Using Non-Causal Convolution*
    :cite:`9747706`.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each ConvEmformer layer.
        ffn_dim (int): hidden layer dimension of each ConvEmformer layer's feedforward network.
        num_layers (int): number of ConvEmformer layers to instantiate.
        segment_length (int): length of each input segment.
        kernel_size (int): size of kernel to use in convolution modules.
        dropout (float, optional): dropout probability. (Default: 0.0)
        ffn_activation (str, optional): activation function to use in feedforward networks.
            Must be one of ("relu", "gelu", "silu"). (Default: "relu")
        left_context_length (int, optional): length of left context. (Default: 0)
        right_context_length (int, optional): length of right context. (Default: 0)
        max_memory_size (int, optional): maximum number of memory elements to use. (Default: 0)
        weight_init_scale_strategy (str or None, optional): per-layer weight initialization scaling
            strategy. Must be one of ("depthwise", "constant", ``None``). (Default: "depthwise")
        tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
        negative_inf (float, optional): value to use for negative infinity in attention weights. (Default: -1e8)
        conv_activation (str, optional): activation function to use in convolution modules.
            Must be one of ("relu", "gelu", "silu"). (Default: "silu")

    Examples:
        >>> conv_emformer = ConvEmformer(80, 4, 1024, 12, 16, 8, right_context_length=4)
        >>> input = torch.rand(10, 200, 80)
        >>> lengths = torch.randint(1, 200, (10,))
        >>> output, lengths = conv_emformer(input, lengths)
        >>> input = torch.rand(4, 20, 80)
        >>> lengths = torch.ones(4) * 20
        >>> output, lengths, states = conv_emformer.infer(input, lengths, None)
    """

    def __init__(self, input_dim: int, num_heads: int, ffn_dim: int, num_layers: int, segment_length: int, kernel_size: int, dropout: float=0.0, ffn_activation: str='relu', left_context_length: int=0, right_context_length: int=0, max_memory_size: int=0, weight_init_scale_strategy: Optional[str]='depthwise', tanh_on_mem: bool=False, negative_inf: float=-100000000.0, conv_activation: str='silu'):
        weight_init_gains = _get_weight_init_gains(weight_init_scale_strategy, num_layers)
        emformer_layers = torch.nn.ModuleList([_ConvEmformerLayer(input_dim, num_heads, ffn_dim, segment_length, kernel_size, dropout=dropout, ffn_activation=ffn_activation, left_context_length=left_context_length, right_context_length=right_context_length, max_memory_size=max_memory_size, weight_init_gain=weight_init_gains[layer_idx], tanh_on_mem=tanh_on_mem, negative_inf=negative_inf, conv_activation=conv_activation) for layer_idx in range(num_layers)])
        super().__init__(emformer_layers, segment_length, left_context_length=left_context_length, right_context_length=right_context_length, max_memory_size=max_memory_size)