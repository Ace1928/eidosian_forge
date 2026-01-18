import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
import numpy as np
from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export
def _get_quant_params(tensor_detail: Mapping[str, Any]) -> Optional[Tuple[float, int]]:
    """Returns first scale and zero point from tensor detail, if present."""
    quant_params = tensor_detail['quantization_parameters']
    if not quant_params:
        return None
    if quant_params['scales'] and quant_params['zero_points']:
        return (quant_params['scales'][0], quant_params['zero_points'][0])
    return None