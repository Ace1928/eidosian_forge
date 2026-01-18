import copy
from typing import Any, Dict, Optional, TypeVar, Union, overload
import warnings
import torch
from torch import Tensor, device, dtype, nn
import torch.nn.functional as F
import bitsandbytes as bnb
from bitsandbytes.autograd._functions import get_tile_inds, undo_layout
from bitsandbytes.functional import QuantState
from bitsandbytes.optim import GlobalOptimManager
from bitsandbytes.utils import OutlierTracer
class StableEmbedding(torch.nn.Embedding):
    """
    Custom embedding layer designed to improve stability during training for NLP tasks by using 32-bit optimizer states. It is designed to reduce gradient variations that can result from quantization. This embedding layer is initialized with Xavier uniform initialization followed by layer normalization.

    Example:

    ```
    # Initialize StableEmbedding layer with vocabulary size 1000, embedding dimension 300
    embedding_layer = StableEmbedding(num_embeddings=1000, embedding_dim=300)

    # Reset embedding parameters
    embedding_layer.reset_parameters()

    # Perform a forward pass with input tensor
    input_tensor = torch.tensor([1, 2, 3])
    output_embedding = embedding_layer(input_tensor)
    ```

    Attributes:
        norm (`torch.nn.LayerNorm`): Layer normalization applied after the embedding.

    Methods:
        reset_parameters(): Reset embedding parameters using Xavier uniform initialization.
        forward(input: Tensor) -> Tensor: Forward pass through the stable embedding layer.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]=None, max_norm: Optional[float]=None, norm_type: float=2.0, scale_grad_by_freq: bool=False, sparse: bool=False, _weight: Optional[Tensor]=None, device=None, dtype=None) -> None:
        """
        Args:
            num_embeddings (`int`):
                The number of unique embeddings (vocabulary size).
            embedding_dim (`int`):
                The dimensionality of the embedding.
            padding_idx (`Optional[int]`):
                Pads the output with zeros at the given index.
            max_norm (`Optional[float]`):
                Renormalizes embeddings to have a maximum L2 norm.
            norm_type (`float`, defaults to `2.0`):
                The p-norm to compute for the `max_norm` option.
            scale_grad_by_freq (`bool`, defaults to `False`):
                Scale gradient by frequency during backpropagation.
            sparse (`bool`, defaults to `False`):
                Computes dense gradients. Set to `True` to compute sparse gradients instead.
            _weight (`Optional[Tensor]`):
                Pretrained embeddings.
        """
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, device, dtype)
        self.norm = torch.nn.LayerNorm(embedding_dim, device=device)
        GlobalOptimManager.get_instance().register_module_override(self, 'weight', {'optim_bits': 32})

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight)
        self._fill_padding_idx_with_zero()
    ' !!! This is a redefinition of _fill_padding_idx_with_zero in torch.nn.Embedding\n        to make the Layer compatible with Pytorch < 1.9.\n        This means that if this changes in future PyTorch releases this need to change too\n        which is cumbersome. However, with this we can ensure compatibility with previous\n        PyTorch releases.\n    '

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        emb = F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        emb = emb.to(torch.get_default_dtype())
        return self.norm(emb).to(self.weight.dtype)