from collections import namedtuple
from typing import Any, Dict, NewType, Optional, Sequence, Tuple, Type
import numpy
import onnx.checker
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx import IR_VERSION, ModelProto, NodeProto
class BackendRep:
    """BackendRep is the handle that a Backend returns after preparing to execute
    a model repeatedly. Users will then pass inputs to the run function of
    BackendRep to retrieve the corresponding results.
    """

    def run(self, inputs: Any, **kwargs: Any) -> Tuple[Any, ...]:
        """Abstract function."""
        return (None,)