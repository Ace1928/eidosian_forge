from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import load
from onnx.defs import onnx_opset_version
from onnx.external_data_helper import ExternalDataInfo, uses_external_data
from onnx.model_container import ModelContainer
from onnx.onnx_pb import (
from onnx.reference.op_run import (
from onnx.reference.ops_optimized import optimized_operators
def _log_arg(self, a: Any) -> Any:
    if isinstance(a, (str, int, float)):
        return a
    if isinstance(a, np.ndarray):
        if self.verbose < 4:
            return f'{a.dtype}:{a.shape} in [{a.min()}, {a.max()}]'
        elements = a.ravel().tolist()
        if len(elements) > 5:
            elements = elements[:5]
            return f'{a.dtype}:{a.shape}:{','.join(map(str, elements))}...'
        return f'{a.dtype}:{a.shape}:{elements}'
    if hasattr(a, 'append'):
        return ', '.join(map(self._log_arg, a))
    return a