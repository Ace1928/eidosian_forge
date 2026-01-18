import collections
import math
import numbers
from typing import Any, Dict as PythonDict, Hashable, List as PythonList, Optional, Sequence, Tuple as PythonTuple, Type
import weakref
from tensorflow.core.function.trace_type import default_types_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
def _has_same_structure(self, other):
    if not isinstance(other, Dict):
        return False
    return self.mapping.keys() == other.mapping.keys()