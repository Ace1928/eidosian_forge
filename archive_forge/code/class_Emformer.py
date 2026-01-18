import math
from typing import List, Optional, Tuple
import torch
class Emformer(_EmformerImpl):
    """Emformer architecture introduced in
    *Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency Streaming Speech Recognition*
    :cite:`shi2021emformer`.

    See Also:
        * :func:`~torchaudio.models.emformer_rnnt_model`,
          :func:`~torchaudio.models.emformer_rnnt_base`: factory functions.
        * :class:`torchaudio.pipelines.RNNTBundle`: ASR pipelines with pretrained model.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Emformer layer.
        ffn_dim (int): hidden layer dimension of each Emformer layer's feedforward network.
        num_layers (int): number of Emformer layers to instantiate.
        segment_length (int): length of each input segment.
        dropout (float, optional): dropout probability. (Default: 0.0)
        activation (str, optional): activation function to use in each Emformer layer's
            feedforward network. Must be one of ("relu", "gelu", "silu"). (Default: "relu")
        left_context_length (int, optional): length of left context. (Default: 0)
        right_context_length (int, optional): length of right context. (Default: 0)
        max_memory_size (int, optional): maximum number of memory elements to use. (Default: 0)
        weight_init_scale_strategy (str or None, optional): per-layer weight initialization scaling
            strategy. Must be one of ("depthwise", "constant", ``None``). (Default: "depthwise")
        tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
        negative_inf (float, optional): value to use for negative infinity in attention weights. (Default: -1e8)

    Examples:
        >>> emformer = Emformer(512, 8, 2048, 20, 4, right_context_length=1)
        >>> input = torch.rand(128, 400, 512)  # batch, num_frames, feature_dim
        >>> lengths = torch.randint(1, 200, (128,))  # batch
        >>> output, lengths = emformer(input, lengths)
        >>> input = torch.rand(128, 5, 512)
        >>> lengths = torch.ones(128) * 5
        >>> output, lengths, states = emformer.infer(input, lengths, None)
    """

    def __init__(self, input_dim: int, num_heads: int, ffn_dim: int, num_layers: int, segment_length: int, dropout: float=0.0, activation: str='relu', left_context_length: int=0, right_context_length: int=0, max_memory_size: int=0, weight_init_scale_strategy: Optional[str]='depthwise', tanh_on_mem: bool=False, negative_inf: float=-100000000.0):
        weight_init_gains = _get_weight_init_gains(weight_init_scale_strategy, num_layers)
        emformer_layers = torch.nn.ModuleList([_EmformerLayer(input_dim, num_heads, ffn_dim, segment_length, dropout=dropout, activation=activation, left_context_length=left_context_length, max_memory_size=max_memory_size, weight_init_gain=weight_init_gains[layer_idx], tanh_on_mem=tanh_on_mem, negative_inf=negative_inf) for layer_idx in range(num_layers)])
        super().__init__(emformer_layers, segment_length, left_context_length=left_context_length, right_context_length=right_context_length, max_memory_size=max_memory_size)