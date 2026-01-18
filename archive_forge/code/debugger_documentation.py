import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
import numpy as np
from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export
Dumps layer statistics into file, in csv format.

    Args:
      file: file, or file-like object to write.
    