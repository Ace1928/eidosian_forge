from __future__ import annotations
from typing import Dict, List
import torch
from torch.fx import Node
from .quantizer import QuantizationAnnotation, Quantizer
def _record_and_validate_annotations(self, gm: torch.fx.GraphModule, quantizer: Quantizer) -> None:
    for n in gm.graph.nodes:
        if 'quantization_annotation' in n.meta:
            if n in self._graph_annotations and id(self._graph_annotations[n]) != id(n.meta['quantization_annotation']):
                raise RuntimeError(f'Quantizer {quantizer.__class__.__name__} has changed annotations on node {n}')
            else:
                self._graph_annotations[n] = n.meta['quantization_annotation']
        elif n in self._graph_annotations:
            raise RuntimeError(f'Quantizer {quantizer.__class__.__name__} has removed annotations on node {n}')