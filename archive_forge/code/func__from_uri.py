import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _from_uri(data: DataType, missing: Optional[FloatCompatible], feature_names: Optional[FeatureNames], feature_types: Optional[FeatureTypes], data_split_mode: DataSplitMode=DataSplitMode.ROW) -> DispatchedDataBackendReturnType:
    _warn_unused_missing(data, missing)
    handle = ctypes.c_void_p()
    data = os.fspath(os.path.expanduser(data))
    args = {'uri': str(data), 'data_split_mode': int(data_split_mode)}
    config = bytes(json.dumps(args), 'utf-8')
    _check_call(_LIB.XGDMatrixCreateFromURI(config, ctypes.byref(handle)))
    return (handle, feature_names, feature_types)