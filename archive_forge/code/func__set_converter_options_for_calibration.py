import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
import numpy as np
from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export
def _set_converter_options_for_calibration(self, converter: TFLiteConverter) -> TFLiteConverter:
    """Verify converter options and set required experimental options."""
    if not converter.optimizations:
        raise ValueError('converter object must set optimizations to lite.Optimize.DEFAULT')
    if not converter.representative_dataset:
        raise ValueError('converter object must set representative_dataset')
    converter.experimental_mlir_quantizer = True
    converter._experimental_calibrate_only = True
    return converter