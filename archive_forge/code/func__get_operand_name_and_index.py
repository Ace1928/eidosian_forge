import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
import numpy as np
from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export
def _get_operand_name_and_index(self, numeric_verify_name: str) -> Tuple[str, int]:
    """Gets the index and name of NumericVerify Op's quantized input tensor.

    Args:
      numeric_verify_name: name of the NumericVerify op's output tensor. It has
        format of `NumericVerify/{quantized_tensor_name}:{quantized_tensor_idx}`

    Returns:
      Tuple of (tensor_name, tensor_idx) for quantized op's output tensor.
    """
    tensor_name, tensor_idx = numeric_verify_name.rsplit(':', 1)
    float_tensor_name = tensor_name[len(_NUMERIC_VERIFY_OP_NAME) + 1:]
    if re.match('\\d', float_tensor_name[-1]):
        float_tensor_name = float_tensor_name[:-1]
    return (float_tensor_name, int(tensor_idx))