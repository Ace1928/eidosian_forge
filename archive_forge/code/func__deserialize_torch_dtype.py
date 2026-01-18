from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def _deserialize_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert the string-encoded `torch_dtype` pipeline argument back to the correct `torch.dtype`
    instance value for applying to a loaded pipeline instance.
    """
    try:
        import torch
    except ImportError as e:
        raise MlflowException('Unable to determine if the value supplied by the argument torch_dtype is valid since torch is not installed.', error_code=INVALID_PARAMETER_VALUE) from e
    if dtype_str.startswith('torch.'):
        dtype_str = dtype_str[6:]
    dtype = getattr(torch, dtype_str, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise MlflowException(f"The value '{dtype_str}' is not a valid torch.dtype", error_code=INVALID_PARAMETER_VALUE)