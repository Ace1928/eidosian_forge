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
class TestShapeInferenceHelper(unittest.TestCase):

    def _make_graph(self, seed_values: Sequence[str | tuple[str, TensorProto.DataType, Any]], nodes: list[NodeProto], value_info: list[ValueInfoProto], initializer: Sequence[TensorProto] | None=None) -> GraphProto:
        if initializer is None:
            initializer = []
        names_in_initializer = {x.name for x in initializer}
        input_value_infos = []
        for seed_value in seed_values:
            if isinstance(seed_value, tuple):
                seed_name, proto_type = seed_value[:2]
                seed_value_info = make_tensor_value_info(*seed_value)
            else:
                seed_name, proto_type = (seed_value, TensorProto.UNDEFINED)
                seed_value_info = make_empty_tensor_value_info(seed_value)
            if seed_name in names_in_initializer:
                input_value_infos.append(seed_value_info)
            else:
                value_info.append(seed_value_info)
                input_value_infos.append(make_tensor_value_info('SEED_' + seed_name, proto_type, ()))
                input_value_infos.append(make_tensor_value_info('UNKNOWN_SHAPE_' + seed_name, TensorProto.INT64, (None,)))
                nodes[:0] = [make_node('Reshape', ['SEED_' + seed_name, 'UNKNOWN_SHAPE_' + seed_name], [seed_name])]
        return helper.make_graph(nodes, 'test', input_value_infos, [], initializer=initializer, value_info=value_info)

    def _inferred(self, graph_or_model: GraphProto | ModelProto, **kwargs: Any) -> ModelProto:
        data_prop = kwargs.pop('data_prop', False)
        if isinstance(graph_or_model, GraphProto):
            kwargs['producer_name'] = 'onnx-test'
            orig_model = helper.make_model(graph_or_model, **kwargs)
        else:
            orig_model = graph_or_model
        inferred_model = onnx.shape_inference.infer_shapes(orig_model, strict_mode=True, data_prop=data_prop)
        checker.check_model(inferred_model)
        return inferred_model

    def _assert_inferred(self, graph_or_model: GraphProto | ModelProto, vis: list[ValueInfoProto], **kwargs: Any) -> None:
        graph = graph_or_model if isinstance(graph_or_model, GraphProto) else graph_or_model.graph
        names_in_vis = {x.name for x in vis}
        vis = [x for x in graph.value_info if x.name not in names_in_vis] + vis
        inferred_model = self._inferred(graph_or_model, **kwargs)
        inferred_vis = list(inferred_model.graph.value_info)
        vis = sorted(vis, key=lambda x: x.name)
        inferred_vis = sorted(inferred_vis, key=lambda x: x.name)
        assert len(vis) == len(inferred_vis)
        for v, inferred_v in zip(vis, inferred_vis):
            self._compare_value_infos(v.type, inferred_v.type)

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

    def skipIf(self, condition, reason):
        if condition:
            pytest.skip(reason)