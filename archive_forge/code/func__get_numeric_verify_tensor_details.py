import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
import numpy as np
from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export
def _get_numeric_verify_tensor_details(self) -> List[str]:
    """Returns all names of all tensors from NumericVerify op."""
    if not self._numeric_verify_tensor_details:
        self._numeric_verify_tensor_details = []
        self._numeric_verify_op_details = {}
        for op_info in self._quant_interpreter._get_ops_details():
            if op_info['op_name'] == _NUMERIC_VERIFY_OP_NAME:
                self._numeric_verify_tensor_details.append(self._quant_interpreter._get_tensor_details(op_info['outputs'][0], subgraph_index=0))
                tensor_name = self._numeric_verify_tensor_details[-1]['name']
                self._numeric_verify_op_details[tensor_name] = op_info
    return self._numeric_verify_tensor_details