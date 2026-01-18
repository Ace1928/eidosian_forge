import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def _test_add_prefix(self, rename_nodes: bool=False, rename_edges: bool=False, rename_inputs: bool=False, rename_outputs: bool=False, rename_initializers: bool=False, rename_value_infos: bool=False, inplace: bool=False) -> None:
    m1 = _load_model(M1_DEF)
    prefix = 'pre/'
    if inplace:
        m2 = ModelProto()
        m2.CopyFrom(m1)
        compose.add_prefix(m2, prefix, rename_nodes=rename_nodes, rename_edges=rename_edges, rename_inputs=rename_inputs, rename_outputs=rename_outputs, rename_initializers=rename_initializers, rename_value_infos=rename_value_infos, inplace=True)
    else:
        m2 = compose.add_prefix(m1, prefix, rename_nodes=rename_nodes, rename_edges=rename_edges, rename_inputs=rename_inputs, rename_outputs=rename_outputs, rename_initializers=rename_initializers, rename_value_infos=rename_value_infos)
    g_in = m1.graph
    g_out = m2.graph
    if rename_edges or rename_inputs or rename_outputs or rename_initializers or rename_value_infos:
        name_mapping = {}
        if rename_edges:
            for n in g_in.node:
                for e in n.input:
                    name_mapping[e] = _prefixed(prefix, e)
                for e in n.output:
                    name_mapping[e] = _prefixed(prefix, e)
        if rename_inputs:
            for elem in g_in.input:
                name_mapping[elem.name] = _prefixed(prefix, elem.name)
        if rename_outputs:
            for elem in g_in.output:
                name_mapping[elem.name] = _prefixed(prefix, elem.name)
        if rename_initializers:
            for init in g_in.initializer:
                name_mapping[init.name] = _prefixed(prefix, init.name)
            for sparse_init in g_in.sparse_initializer:
                name_mapping[sparse_init.values.name] = _prefixed(prefix, sparse_init.values.name)
                name_mapping[sparse_init.indices.name] = _prefixed(prefix, sparse_init.indices.name)
        if rename_value_infos:
            for value_info in g_in.output:
                name_mapping[value_info.name] = _prefixed(prefix, value_info.name)
        for n1, n0 in zip(g_out.node, g_in.node):
            for e1, e0 in zip(n1.input, n0.input):
                self.assertEqual(name_mapping.get(e0, e0), e1)
            for e1, e0 in zip(n1.output, n0.output):
                self.assertEqual(name_mapping.get(e0, e0), e1)
        for i1, i0 in zip(g_out.input, g_in.input):
            self.assertEqual(name_mapping.get(i0.name, i0.name), i1.name)
        for o1, o0 in zip(g_out.output, g_in.output):
            self.assertEqual(name_mapping.get(o0.name, o0.name), o1.name)
        for init1, init0 in zip(g_out.initializer, g_in.initializer):
            self.assertEqual(name_mapping.get(init0.name, init0.name), init1.name)
        for sparse_init1, sparse_init0 in zip(g_out.sparse_initializer, g_in.sparse_initializer):
            self.assertEqual(name_mapping.get(sparse_init0.values.name, sparse_init0.values.name), sparse_init1.values.name)
            self.assertEqual(name_mapping.get(sparse_init0.indices.name, sparse_init0.indices.name), sparse_init1.indices.name)
        for vi1, vi0 in zip(g_out.value_info, g_in.value_info):
            self.assertEqual(name_mapping.get(vi0.name, vi0.name), vi1.name)
        if rename_nodes:
            for n1, n0 in zip(g_out.node, g_in.node):
                self.assertEqual(_prefixed(prefix, n0.name), n1.name)