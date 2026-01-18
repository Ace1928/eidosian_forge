import json
import os
import re
import sys
import numpy as np
def GenerateGraph(subgraph_idx, g, opcode_mapper):
    """Produces the HTML required to have a d3 visualization of the dag."""

    def TensorName(idx):
        return 't%d' % idx

    def OpName(idx):
        return 'o%d' % idx
    edges = []
    nodes = []
    first = {}
    second = {}
    pixel_mult = 200
    width_mult = 170
    for op_index, op in enumerate(g['operators'] or []):
        if op['inputs'] is not None:
            for tensor_input_position, tensor_index in enumerate(op['inputs']):
                if tensor_index not in first:
                    first[tensor_index] = ((op_index - 0.5 + 1) * pixel_mult, (tensor_input_position + 1) * width_mult)
                edges.append({'source': TensorName(tensor_index), 'target': OpName(op_index)})
        if op['outputs'] is not None:
            for tensor_output_position, tensor_index in enumerate(op['outputs']):
                if tensor_index not in second:
                    second[tensor_index] = ((op_index + 0.5 + 1) * pixel_mult, (tensor_output_position + 1) * width_mult)
                edges.append({'target': TensorName(tensor_index), 'source': OpName(op_index)})
        nodes.append({'id': OpName(op_index), 'name': opcode_mapper(op['opcode_index']), 'group': 2, 'x': pixel_mult, 'y': (op_index + 1) * pixel_mult})
    for tensor_index, tensor in enumerate(g['tensors']):
        initial_y = first[tensor_index] if tensor_index in first else second[tensor_index] if tensor_index in second else (0, 0)
        nodes.append({'id': TensorName(tensor_index), 'name': '%r (%d)' % (getattr(tensor, 'shape', []), tensor_index), 'group': 1, 'x': initial_y[1], 'y': initial_y[0]})
    graph_str = json.dumps({'nodes': nodes, 'edges': edges})
    html = _D3_HTML_TEMPLATE % (graph_str, subgraph_idx)
    return html