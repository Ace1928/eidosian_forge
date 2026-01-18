from __future__ import annotations
import copy
from typing import List, Set
import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.quantizer.quantizer import (
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
class EmbeddingQuantizer(Quantizer):

    def __init__(self):
        super().__init__()

    @classmethod
    def get_supported_quantization_configs(cls) -> List[QuantizationConfig]:
        op_configs: Set[QuantizationConfig] = set({})
        for spec, _ in cls.get_supported_operators():
            op_configs.add(spec)
        return list(op_configs)

    @classmethod
    def get_supported_operator_for_quantization_config(cls, quantization_config: QuantizationConfig) -> List[OperatorPatternType]:
        for config, ops in cls.get_supported_operators():
            if config == quantization_config:
                return ops
        return []

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        self._annotate_embedding_ops(model.graph)
        return model

    def _annotate_embedding_ops(self, graph: torch.fx.Graph) -> None:
        embedding_config: OperatorConfig = get_embedding_operators_config()
        for node in graph.nodes:
            if node.op == 'call_function' and node.target == torch.ops.aten.embedding.default:
                if embedding_config.config.weight is None:
                    raise ValueError('Embedding config must have a valid weight quantization spec.')
                node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={node.args[0]: embedding_config.config.weight})

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return [get_embedding_operators_config()]