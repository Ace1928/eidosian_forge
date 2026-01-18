import warnings
from typing import Any, Dict, NamedTuple, Union, cast
import numpy as np
from onnx import OptionalProto, SequenceProto, TensorProto
class TensorDtypeMap(NamedTuple):
    np_dtype: np.dtype
    storage_dtype: int
    name: str