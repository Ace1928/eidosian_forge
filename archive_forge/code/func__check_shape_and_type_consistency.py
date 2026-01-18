from typing import Optional, Tuple
import torch
from torch import Tensor
def _check_shape_and_type_consistency(preds: Tensor, target: Tensor) -> None:
    """Check shape and type consistency of input vectors.

    Args:
        preds:
            Logits or a unnormalized score assigned to each token in a sequence with shape [batch_size, seq_len,
            vocab_size]. Scores will be normalized internally using softmax.
        target:
            Ground truth values with a shape [batch_size, seq_len].

    Raises:
        ValueError:
            If ``preds`` tensor has no 3 dimensions.
        ValueError:
            If ``target`` tensor has no 2 dimensions.
        ValueError:
            If the first two dimensions of ``preds`` and ``target`` do not equal.
        TypeError:
            If ``preds`` dtype is not one of ``(torch.float16, torch.float32, torch.float64)``
        TypeError:
            If ``target`` is not of a type LongTensor (torch.int64)

    """
    if len(preds.shape) != 3:
        raise ValueError(f'Input tensor `preds` is expected to have 3 dimensions, [batch_size, seq_len, vocab_size], but got {len(preds.shape)}.')
    if len(target.shape) != 2:
        raise ValueError(f'Input tensor `target` is expected to have 2 dimensions, [batch_size, seq_len], but got {len(target.shape)}.')
    if preds.shape[:2] != target.shape:
        raise ValueError(f'Input tensors `preds` and `target` are expected to have equaling first two dimensions, [batch_size, seq_len], but got {preds.shape[:2]} and {target.shape}.')
    if not preds.is_floating_point():
        raise TypeError(f'Input tensor `preds` is expected to be of floating point type but got {preds.dtype}.')
    if target.dtype != torch.int64:
        raise TypeError(f'Input tensor `target` is expected to be of a type {torch.int64} but got {target.dtype}.')