from typing import List
from torch.ao.quantization.pt2e.utils import _is_sym_size_node
from torch.ao.quantization.quantizer.quantizer import QuantizationAnnotation
from torch.fx import Node
def _annotate_input_qspec_map(node: Node, input_node: Node, qspec):
    quantization_annotation = node.meta.get('quantization_annotation', QuantizationAnnotation())
    if quantization_annotation.input_qspec_map is None:
        quantization_annotation.input_qspec_map = {}
    quantization_annotation.input_qspec_map[input_node] = qspec
    node.meta['quantization_annotation'] = quantization_annotation