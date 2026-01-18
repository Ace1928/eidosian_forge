import warnings
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch import Tensor
from torch.masked import as_masked_tensor, is_masked_tensor, MaskedTensor
from . import _docs
from torch._prims_common import corresponding_real_dtype
from torch import sym_float
def _apply_docstring_templates(func):
    """Decorator that applies docstring templates to function docstring
    and returns the function instance.
    """
    doc_string = getattr(_docs, f'{func.__name__}_docstring', None)
    if doc_string is None:
        warnings.warn(f'No documentation string available for {func.__name__}. PyTorch team should run `python tools/update_masked_docs.py` to generate the missing docstrings.')
    else:
        func.__doc__ = doc_string
    __all__.append(func.__name__)
    return func