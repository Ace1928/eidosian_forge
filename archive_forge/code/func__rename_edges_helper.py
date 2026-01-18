import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.case.utils import import_recursive
from onnx.onnx_pb import (
def _rename_edges_helper(internal_node: NodeProto, rename_helper: Callable[[str], str], attribute_map: Dict[str, AttributeProto], prefix: str) -> NodeProto:
    new_node = NodeProto()
    new_node.CopyFrom(internal_node)
    new_node.ClearField('input')
    new_node.ClearField('output')
    new_node.ClearField('attribute')
    for internal_name in internal_node.input:
        new_node.input.append(rename_helper(internal_name))
    for internal_name in internal_node.output:
        new_node.output.append(rename_helper(internal_name))
    for attr in internal_node.attribute:
        if attr.HasField('ref_attr_name'):
            if attr.ref_attr_name in attribute_map:
                new_attr = AttributeProto()
                new_attr.CopyFrom(attribute_map[attr.ref_attr_name])
                new_attr.name = attr.name
                new_node.attribute.extend([new_attr])
        else:
            new_attr = AttributeProto()
            new_attr.CopyFrom(attr)
            if attr.type == AttributeProto.GRAPH:
                new_graph = new_attr.g
                sg_rename = {}
                for in_desc in new_graph.input:
                    sg_rename[in_desc.name] = in_desc.name = prefix + in_desc.name
                for out_desc in new_graph.output:
                    sg_rename[out_desc.name] = out_desc.name = prefix + out_desc.name
                for init_desc in new_graph.initializer:
                    sg_rename[init_desc.name] = init_desc.name = prefix + init_desc.name
                for sparse_init_desc in new_graph.sparse_initializer:
                    sg_rename[sparse_init_desc.values.name] = sparse_init_desc.values.name = prefix + sparse_init_desc.values.name
                for sparse_init_desc in new_graph.sparse_initializer:
                    sg_rename[sparse_init_desc.indices.name] = sparse_init_desc.indices.name = prefix + sparse_init_desc.indices.name

                def subgraph_rename_helper(name: str) -> Any:
                    if name in sg_rename:
                        return sg_rename[name]
                    return rename_helper(name)
                new_nodes = [_rename_edges_helper(node_desc, subgraph_rename_helper, attribute_map, prefix) for node_desc in new_graph.node]
                new_graph.ClearField('node')
                new_graph.node.extend(new_nodes)
            new_node.attribute.extend([new_attr])
    return new_node