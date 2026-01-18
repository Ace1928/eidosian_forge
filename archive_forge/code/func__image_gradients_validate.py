from typing import Tuple
import torch
from torch import Tensor
def _image_gradients_validate(img: Tensor) -> None:
    """Validate whether img is a 4D torch Tensor."""
    if not isinstance(img, Tensor):
        raise TypeError(f'The `img` expects a value of <Tensor> type but got {type(img)}')
    if img.ndim != 4:
        raise RuntimeError(f'The `img` expects a 4D tensor but got {img.ndim}D tensor')