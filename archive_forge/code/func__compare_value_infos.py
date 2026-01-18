from __future__ import annotations
import itertools
import unittest
from typing import Any, Sequence
import numpy as np
import pytest
from parameterized import parameterized
import onnx.shape_inference
from onnx import (
from onnx.defs import (
from onnx.helper import (
from onnx.parser import parse_graph
def _compare_value_infos(self, vi_type: TypeProto, inferred_vi_type: TypeProto) -> None:
    if vi_type.HasField('tensor_type'):
        assert inferred_vi_type.HasField('tensor_type')
        assert vi_type.tensor_type.HasField('elem_type')
        assert inferred_vi_type.tensor_type.HasField('elem_type')
        assert vi_type.tensor_type.elem_type == inferred_vi_type.tensor_type.elem_type
        assert vi_type.tensor_type.HasField('shape') == inferred_vi_type.tensor_type.HasField('shape')
        if vi_type.tensor_type.HasField('shape'):
            assert len(vi_type.tensor_type.shape.dim) == len(inferred_vi_type.tensor_type.shape.dim)
            for dim_i, dim in enumerate(vi_type.tensor_type.shape.dim):
                inferred_dim = inferred_vi_type.tensor_type.shape.dim[dim_i]
                if dim.dim_param:
                    assert dim.dim_param == inferred_dim.dim_param, f'\n{vi_type}\n{inferred_vi_type}\n'
                else:
                    assert dim.dim_value == inferred_dim.dim_value, f'\n{vi_type}\n{inferred_vi_type}\n'
    elif vi_type.HasField('sequence_type'):
        assert inferred_vi_type.HasField('sequence_type')
        vi = vi_type.sequence_type.elem_type
        inferred_vi = inferred_vi_type.sequence_type.elem_type
        self._compare_value_infos(vi, inferred_vi)
    elif vi_type.HasField('optional_type'):
        assert inferred_vi_type.HasField('optional_type')
        vi = vi_type.optional_type.elem_type
        inferred_vi = inferred_vi_type.optional_type.elem_type
        self._compare_value_infos(vi, inferred_vi)
    elif vi_type.HasField('map_type'):
        assert inferred_vi_type.HasField('map_type')
        assert vi_type.map_type.key_type == vi_type.map_type.key_type
        self._compare_value_infos(vi_type.map_type.value_type, inferred_vi_type.map_type.value_type)
    elif vi_type == onnx.TypeProto():
        assert inferred_vi_type == onnx.TypeProto()
    else:
        raise NotImplementedError('Unrecognized value info type in _compare_value_infos: ', str(vi_type))