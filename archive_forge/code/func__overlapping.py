from typing import Dict, List, MutableMapping, Optional, Set, Tuple
from onnx import GraphProto, ModelProto, TensorProto, checker, helper, utils
def _overlapping(c1: List[str], c2: List[str]) -> List[str]:
    return list(set(c1) & set(c2))