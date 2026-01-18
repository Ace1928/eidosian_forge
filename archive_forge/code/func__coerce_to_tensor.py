import operator
import torch
from . import _dtypes_impl
def _coerce_to_tensor(obj, dtype=None, copy=False, ndmin=0):
    """The core logic of the array(...) function.

    Parameters
    ----------
    obj : tensor_like
        The thing to coerce
    dtype : torch.dtype object or None
        Coerce to this torch dtype
    copy : bool
        Copy or not
    ndmin : int
        The results as least this many dimensions
    is_weak : bool
        Whether obj is a weakly typed python scalar.

    Returns
    -------
    tensor : torch.Tensor
        a tensor object with requested dtype, ndim and copy semantics.

    Notes
    -----
    This is almost a "tensor_like" coersion function. Does not handle wrapper
    ndarrays (those should be handled in the ndarray-aware layer prior to
    invoking this function).
    """
    if isinstance(obj, torch.Tensor):
        tensor = obj
    else:
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(_dtypes_impl.get_default_dtype_for(torch.float32))
        try:
            tensor = _try_convert_to_tensor(obj)
        finally:
            torch.set_default_dtype(default_dtype)
    tensor = cast_if_needed(tensor, dtype)
    ndim_extra = ndmin - tensor.ndim
    if ndim_extra > 0:
        tensor = tensor.view((1,) * ndim_extra + tensor.shape)
    if copy:
        tensor = tensor.clone()
    return tensor