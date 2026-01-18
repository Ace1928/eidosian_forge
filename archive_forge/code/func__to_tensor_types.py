import unittest
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
from onnx import TensorProto, TypeProto
from onnx.checker import ValidationError
from onnx.defs import OpSchema, get_all_schemas_with_history, get_schema
from onnx.helper import (
from onnx.numpy_helper import from_array
from onnx.shape_inference import InferenceError, infer_node_outputs
def _to_tensor_types(tensor_types: Dict[str, Tuple[int, Tuple[Union[int, str, None], ...]]]) -> Dict[str, TypeProto]:
    return {key: make_tensor_type_proto(*value) for key, value in tensor_types.items()}