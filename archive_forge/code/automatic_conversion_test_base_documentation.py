from __future__ import annotations
import string
import unittest
from typing import Any, List, Sequence, cast
import onnx
from onnx import TensorProto, ValueInfoProto, helper, shape_inference, version_converter
Test conversion.

        Args:
            op: A string representing the name of the operator to test.
            from_opset: An integer representing the lowest opset version to convert.
            input_shapes: A sequence of tuples or strings representing the shapes of the input tensors.
                The default value is ((3, 4, 5),).
            output_shapes: A sequence of tuples representing the shapes of the output tensors.
                The default value is ((3, 4, 5),).
            input_types: An optional sequence of types representing the data types of the input tensors.
            output_types: An optional sequence of types representing the data types of the output tensors.
            initializer: A sequence of values representing the initial values of the input tensors.
            attrs: An optional dictionary of attributes for the operator.
            seq_inputs: A sequence of integers representing the indices of the input tensors that are sequences.
            seq_outputs: A sequence of integers representing the indices of the output tensors that are sequences.
            optional_inputs: A sequence of integers representing the indices of the input tensors that are optional.
            optional_outputs: A sequence of integers representing the indices of the output tensors that are optional.
            is_upgrade: A boolean value indicating whether to run the version converter from from_opset to
                the most recent opset version (True) or from the most recent opset version to from_opset (False).
                The default value is True. In both cases, runs checker and shape inference on the final model.
        