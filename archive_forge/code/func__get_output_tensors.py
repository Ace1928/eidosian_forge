import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
import numpy as np
from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export
def _get_output_tensors(self, interpreter: _interpreter.Interpreter) -> List[np.ndarray]:
    """Returns output tensors of given TFLite model Interpreter.

    Args:
      interpreter: a tf.lite.Interpreter object with allocated tensors.

    Returns:
      a list of numpy arrays representing output tensor results.
    """
    outputs = []
    for output_detail in interpreter.get_output_details():
        tensor = interpreter.get_tensor(output_detail['index'])
        if output_detail['dtype'] == np.int8:
            quant_params = _get_quant_params(output_detail)
            if quant_params:
                scale, zero_point = quant_params
                tensor = ((tensor.astype(np.float32) - zero_point) * scale).astype(np.float32)
        outputs.append(tensor)
    return outputs