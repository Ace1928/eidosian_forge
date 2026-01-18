import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _from_tuple(data: Sequence, missing: FloatCompatible, n_threads: int, feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes]) -> DispatchedDataBackendReturnType:
    return _from_list(data, missing, n_threads, feature_names, feature_types)