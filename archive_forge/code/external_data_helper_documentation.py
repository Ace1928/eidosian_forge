import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
Serializes data for all the tensors which have data location set to TensorProto.External.

    Note: This function also strips basepath information from all tensors' external_data fields.

    Arguments:
        model (ModelProto): Model object which is the source of tensors to serialize.
        filepath: System path to the directory which should be treated as base path for external data.

    Returns:
        ModelProto: The modified model object.
    