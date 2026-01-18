from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def _extract_torch_dtype_if_set(pipeline) -> Optional[torch.dtype]:
    """
    Extract the torch datatype argument if set and return as a string encoded value.
    """
    try:
        import torch
    except ImportError:
        return None
    model_dtype = pipeline.model.dtype if hasattr(pipeline.model, 'dtype') else None
    return model_dtype if isinstance(model_dtype, torch.dtype) else None