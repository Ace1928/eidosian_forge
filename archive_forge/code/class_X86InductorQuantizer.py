import copy
import functools
import itertools
import operator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import torch
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import (
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer.quantizer import (
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import (
class X86InductorQuantizer(Quantizer):
    supported_config_and_operators = _get_supported_config_and_operators()

    def __init__(self):
        super().__init__()
        self.global_config: QuantizationConfig = None
        self.operator_type_config: Dict[str, Optional[QuantizationConfig]] = {}

    @classmethod
    def get_supported_quantization_configs(cls) -> List[QuantizationConfig]:
        op_configs: Set[QuantizationConfig] = set({})
        for spec, _ in cls.supported_config_and_operators:
            op_configs.add(spec)
        return list(op_configs)

    @classmethod
    def get_supported_operator_for_quantization_config(cls, quantization_config: Optional[QuantizationConfig]) -> List[OperatorPatternType]:
        if quantization_config is None:
            all_ops = []
            for _, ops in cls.supported_config_and_operators:
                all_ops.extend(ops)
            return all_ops
        for config, ops in cls.supported_config_and_operators:
            if config == quantization_config:
                return ops
        return []

    def set_global(self, quantization_config: QuantizationConfig):
        self.global_config = quantization_config
        return self

    def set_config_for_operator_type(self, operator_type: str, quantization_config: QuantizationConfig):
        self.operator_type_config[operator_type] = quantization_config
        return self

    def _annotate_conv_node_helper(self, conv_node: torch.fx.Node, annotate_output: bool, quantization_config: QuantizationConfig) -> None:
        """Helper function to annotate the conv node"""
        input_qspec_map = {}
        input_node = conv_node.args[0]
        assert isinstance(input_node, Node)
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        weight_node = conv_node.args[1]
        assert isinstance(weight_node, Node)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        bias_node = None if len(conv_node.args) == 2 else conv_node.args[2]
        if isinstance(bias_node, Node):
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        if annotate_output:
            conv_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True, _is_output_of_quantized_pattern=True)
        else:
            conv_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True)

    def _annotate_linear_node_helper(self, linear_node: torch.fx.Node, annotate_output: bool, quantization_config: QuantizationConfig) -> None:
        """Helper function to annotate the linear node"""
        input_qspec_map = {}
        assert linear_node.target in (torch.ops.aten.linear.default,)
        has_bias = len(linear_node.args) == 3
        input_index = 0
        weight_index = 1
        bias_index = 2
        input_node = linear_node.args[input_index]
        assert isinstance(input_node, Node)
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        weight_node = linear_node.args[weight_index]
        assert isinstance(weight_node, Node)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        bias_node = linear_node.args[bias_index] if has_bias else None
        if isinstance(bias_node, Node):
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        if annotate_output:
            linear_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True, _is_output_of_quantized_pattern=True)
        else:
            linear_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True)

    def _get_output_nodes_of_partitions(self, partition_list: List[SourcePartition]) -> List[torch.fx.Node]:
        """Helper function to get the output node list from partition list"""
        output_node_list = []
        for partition in partition_list:
            if len(partition.output_nodes) > 1:
                raise ValueError('Input partition has more than one output node')
            output_node = partition.output_nodes[0]
            assert isinstance(output_node, Node)
            output_node_list.append(output_node)
        if len(output_node_list) != len(partition_list):
            raise ValueError('length of output_node_list should equal to length of partition_list')
        return output_node_list

    def _get_input_idx_for_binary_node(self, conv_gemm_node: torch.fx.Node, binary_node: torch.fx.Node):
        """Helper function to check conv_gemm and extra input node index
        for binary node fused with conv_gemm.
        """
        conv_gemm_node_idx = None
        extra_input_node_idx = None
        if binary_node.args[0].op == 'call_function' and binary_node.args[0] == conv_gemm_node:
            conv_gemm_node_idx = 0
            extra_input_node_idx = 1
        elif binary_node.args[1].op == 'call_function' and binary_node.args[1] == conv_gemm_node:
            conv_gemm_node_idx = 1
            extra_input_node_idx = 0
        extra_input_node = binary_node.args[extra_input_node_idx]
        assert isinstance(extra_input_node, Node)
        return (conv_gemm_node_idx, extra_input_node_idx)

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        model = self._annotate_for_static_quantization_config(model)
        return model

    def _annotate_for_static_quantization_config(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """
        High-level description of quantization recipe for X86 Inductor Backend:
        Step 1: Apply quantization recipe for fusion patterns of conv/linear to enable int8 data type actively.
        Step 2: Propagate quantization annotation for patterns besides conv/linear. Go through the pattern in model
        from start to the end. If a pattern supports computation with int8 data type and inputs connected to
        quantized patterns, annotate its inputs as quantized pattern.
        Step 3: Since in step 2, we only annotate the inputs of quantized pattern. For some quantized patterns,
        such as maxpool2d, which only supports output with int8 data type when the input is with int8 data type,
        we need to annotate the output of this pattern.
        """
        config = self.global_config
        if config.is_qat:
            self._annotate_qat_conv2d_fusion_pattern(model, config)
        self._annotate_conv2d_fusion_pattern(model, config)
        for node in model.graph.nodes:
            self._annotation_propagation_quantizable_pattern(node, config)
        for node in model.graph.nodes:
            self._annotate_output_for_int8_in_int8_out_pattern(node, config)
        return model

    def _annotate_qat_conv2d_fusion_pattern(self, model: torch.fx.GraphModule, config: QuantizationConfig):
        self._annotate_qat_conv2d_bn_binary_unary(model, config)
        self._annotate_qat_conv2d_bn_binary(model, config)
        self._annotate_qat_conv2d_bn_unary(model, config)
        self._annotate_qat_conv2d_bn(model, config)

    def _annotate_qat_conv2d_bn_binary_unary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        fused_partitions = find_sequential_partitions(gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d, operator.add, torch.nn.ReLU])
        for fused_partition in fused_partitions:
            conv_partition, bn_partition, binary_partition, unary_partition = fused_partition
            conv_node, bn_output_node, binary_node, unary_node = self._get_output_nodes_of_partitions([conv_partition, bn_partition, binary_partition, unary_partition])
            if len(bn_output_node.users) != 1:
                continue
            bn_output_node_idx, extra_input_node_idx = self._get_input_idx_for_binary_node(bn_output_node, binary_node)
            if bn_output_node_idx is None or extra_input_node_idx is None:
                continue
            if bn_output_node != binary_node.args[bn_output_node_idx]:
                raise ValueError(f"{bn_output_node} doesn't match input of binary node")
            extra_input_node = binary_node.args[extra_input_node_idx]
            if conv_node.op != 'call_function' or conv_node.target != torch.ops.aten.conv2d.default:
                continue
            if _is_annotated([unary_node, binary_node, bn_output_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(quantization_config)
            binary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=binary_node_input_qspec_map, _annotated=True)
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(output_qspec=get_output_act_qspec(quantization_config), _annotated=True, _is_output_of_quantized_pattern=True)
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            nodes_to_mark_annotated.extend(list(binary_partition.nodes))
            nodes_to_mark_annotated.extend(list(unary_partition.nodes))
            _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_qat_conv2d_bn_binary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        fused_partitions = find_sequential_partitions(gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d, operator.add])
        for fused_partition in fused_partitions:
            conv_partition, bn_partition, binary_partition = fused_partition
            conv_node, bn_output_node, binary_node = self._get_output_nodes_of_partitions([conv_partition, bn_partition, binary_partition])
            if len(bn_output_node.users) != 1:
                continue
            bn_output_node_idx, extra_input_node_idx = self._get_input_idx_for_binary_node(bn_output_node, binary_node)
            if bn_output_node_idx is None or extra_input_node_idx is None:
                continue
            if bn_output_node != binary_node.args[bn_output_node_idx]:
                raise ValueError(f"{bn_output_node} doesn't match input of binary node")
            extra_input_node = binary_node.args[extra_input_node_idx]
            if conv_node.op != 'call_function' or conv_node.target != torch.ops.aten.conv2d.default:
                continue
            if _is_annotated([binary_node, bn_output_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(quantization_config)
            binary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=binary_node_input_qspec_map, output_qspec=get_output_act_qspec(quantization_config), _annotated=True, _is_output_of_quantized_pattern=True)
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            nodes_to_mark_annotated.extend(list(binary_partition.nodes))
            _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_qat_conv2d_bn_unary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        fused_partitions = []
        unary_patterns = [[torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU], [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Hardtanh], [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU6]]
        for unary_pattern in unary_patterns:
            partitions = find_sequential_partitions(gm, unary_pattern)
            if partitions:
                fused_partitions.extend(partitions)
        for fused_partition in fused_partitions:
            conv_partition, bn_partition, unary_partition = fused_partition
            conv_node, bn_output_node, unary_node = self._get_output_nodes_of_partitions([conv_partition, bn_partition, unary_partition])
            if conv_node.op != 'call_function' or conv_node.target != torch.ops.aten.conv2d.default:
                continue
            if _is_annotated([unary_node, bn_output_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(output_qspec=get_output_act_qspec(quantization_config), _annotated=True, _is_output_of_quantized_pattern=True)
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            nodes_to_mark_annotated.extend(list(unary_partition.nodes))
            _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_qat_conv2d_bn(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        fused_partitions = find_sequential_partitions(gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d])
        for fused_partition in fused_partitions:
            conv_partition, bn_partition = fused_partition
            conv_node, bn_output_node = self._get_output_nodes_of_partitions([conv_partition, bn_partition])
            if conv_node.op != 'call_function' or conv_node.target != torch.ops.aten.conv2d.default:
                continue
            if _is_annotated([bn_output_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            bn_output_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(output_qspec=get_output_act_qspec(quantization_config), _annotated=True, _is_output_of_quantized_pattern=True)
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_conv2d_fusion_pattern(self, model: torch.fx.GraphModule, config: QuantizationConfig):
        self._annotate_conv2d_binary_unary(model, config)
        self._annotate_conv2d_binary(model, config)
        self._annotate_conv2d_unary(model, config)
        self._annotate_conv2d(model, config)
        self._annotate_linear_unary(model, config)
        self._annotate_linear(model, config)

    def _annotate_conv2d_binary_unary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        fused_partitions = find_sequential_partitions(gm, [torch.nn.Conv2d, operator.add, torch.nn.ReLU])
        for fused_partition in fused_partitions:
            conv_partition, binary_partition, unary_partition = fused_partition
            conv_node, binary_node, unary_node = self._get_output_nodes_of_partitions([conv_partition, binary_partition, unary_partition])
            if len(conv_node.users) != 1:
                continue
            conv_node_idx, extra_input_node_idx = self._get_input_idx_for_binary_node(conv_node, binary_node)
            if conv_node_idx is None or extra_input_node_idx is None:
                continue
            if conv_node != binary_node.args[conv_node_idx]:
                raise ValueError(f"{conv_node} doesn't match input of binary node")
            extra_input_node = binary_node.args[extra_input_node_idx]
            if conv_node.op != 'call_function' or conv_node.target != torch.ops.aten.conv2d.default:
                continue
            if _is_annotated([unary_node, binary_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(quantization_config)
            binary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=binary_node_input_qspec_map, _annotated=True)
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(_annotated=True, _is_output_of_quantized_pattern=True)

    def _annotate_conv2d_binary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        fused_partitions = find_sequential_partitions(gm, [torch.nn.Conv2d, operator.add])
        for fused_partition in fused_partitions:
            conv_partition, binary_partition = fused_partition
            conv_node, binary_node = self._get_output_nodes_of_partitions([conv_partition, binary_partition])
            if len(conv_node.users) != 1:
                continue
            conv_node_idx, extra_input_node_idx = self._get_input_idx_for_binary_node(conv_node, binary_node)
            if conv_node_idx is None or extra_input_node_idx is None:
                continue
            if conv_node != binary_node.args[conv_node_idx]:
                raise ValueError(f"{conv_node} doesn't match input of binary node")
            extra_input_node = binary_node.args[extra_input_node_idx]
            assert isinstance(conv_node, Node)
            if conv_node.op != 'call_function' or conv_node.target != torch.ops.aten.conv2d.default:
                continue
            if _is_annotated([binary_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(quantization_config)
            binary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=binary_node_input_qspec_map, _annotated=True, _is_output_of_quantized_pattern=True)

    def _annotate_conv2d_unary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        fused_partitions = []
        unary_patterns = [[torch.nn.Conv2d, torch.nn.ReLU], [torch.nn.Conv2d, torch.nn.Hardtanh], [torch.nn.Conv2d, torch.nn.ReLU6]]
        for unary_pattern in unary_patterns:
            partitions = find_sequential_partitions(gm, unary_pattern)
            if partitions:
                fused_partitions.extend(partitions)
        for fused_partition in fused_partitions:
            conv_partition, unary_partition = fused_partition
            conv_node, unary_node = self._get_output_nodes_of_partitions([conv_partition, unary_partition])
            if conv_node.op != 'call_function' or conv_node.target != torch.ops.aten.conv2d.default:
                continue
            if _is_annotated([unary_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(_annotated=True, _is_output_of_quantized_pattern=True)

    def _annotate_conv2d(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        conv_partitions = get_source_partitions(gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d])
        conv_partitions = list(itertools.chain(*conv_partitions.values()))
        for conv_partition in conv_partitions:
            if len(conv_partition.output_nodes) > 1:
                raise ValueError('conv partition has more than one output node')
            conv_node = conv_partition.output_nodes[0]
            if conv_node.op != 'call_function' or conv_node.target != torch.ops.aten.conv2d.default:
                raise ValueError(f'{conv_node} is not an aten conv2d operator')
            if _is_annotated([conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, True, quantization_config)

    def _annotate_maxpool2d(self, node: Node, quantization_config: QuantizationConfig) -> None:
        if node.target is not torch.ops.aten.max_pool2d.default:
            return
        maxpool_node = node
        if _is_any_annotated([maxpool_node]):
            return
        input_node = maxpool_node.args[0]
        assert isinstance(input_node, Node)
        input_qspec_map = {}
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        maxpool_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True, _is_output_of_quantized_pattern=True)

    def _annotate_cat(self, node: Node, quantization_config: QuantizationConfig) -> None:
        cat_node = node
        input_nodes = cat_node.args[0]
        assert isinstance(input_nodes, Sequence)
        first_input_node = input_nodes[0]
        input_qspec_map = {}
        assert isinstance(first_input_node, Node)
        assert isinstance(cat_node, Node)
        input_qspec_map[first_input_node] = get_input_act_qspec(quantization_config)
        share_qparams_with_input_act0_qspec = SharedQuantizationSpec((first_input_node, cat_node))
        for input_node in input_nodes[1:]:
            if input_node not in input_qspec_map:
                assert isinstance(input_node, Node)
                input_qspec_map[input_node] = share_qparams_with_input_act0_qspec
        cat_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True, _is_output_of_quantized_pattern=True)

    def _annotation_propagation_quantizable_pattern(self, node: Node, quantization_config: QuantizationConfig) -> None:
        if node.target in quantizable_ops_pt2e and (not _is_any_annotated([node])) and (node.op == 'call_function'):

            def is_all_inputs_connected_to_quantized_op(input_nodes):
                for input_node in input_nodes:
                    if not _is_quantized_op_pt2e(input_node):
                        return False
                return True
            if node.target is torch.ops.aten.max_pool2d.default:
                input_nodes_to_check = [node.all_input_nodes[0]]
                if not is_all_inputs_connected_to_quantized_op(input_nodes_to_check):
                    return
                self._annotate_maxpool2d(node, quantization_config)
                return
            elif node.target is torch.ops.aten.cat.default:
                input_nodes_to_check = node.all_input_nodes
                if not is_all_inputs_connected_to_quantized_op(input_nodes_to_check):
                    return
                self._annotate_cat(node, quantization_config)
            else:
                input_node = node.all_input_nodes[0]
                if not is_all_inputs_connected_to_quantized_op([input_node]):
                    return
                input_qspec_map = {}
                input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
                node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(input_qspec_map=input_qspec_map, _annotated=True, _is_output_of_quantized_pattern=True)
        return

    def _annotate_output_share_observer_as_input(self, input_node: Node, source_node: Node):
        source_node_quantization_annotation = source_node.meta[QUANT_ANNOTATION_KEY] if QUANT_ANNOTATION_KEY in source_node.meta else None
        if source_node_quantization_annotation and source_node_quantization_annotation._is_output_of_quantized_pattern:
            edge_or_node = (input_node, source_node)
            source_node_quantization_annotation.output_qspec = SharedQuantizationSpec(edge_or_node)
        return

    def _annotate_output_for_int8_in_int8_out_pattern(self, node: Node, quantization_config: QuantizationConfig) -> None:
        """
        Check and insert observer at output of node in int8_in_int8_out_ops_pt2e if needed.
        Recipe refers to https://github.com/intel/intel-extension-for-pytorch/blob/
        90d19323d96afc53fcc22ba5a7bb3fb07fdd6c1c/intel_extension_for_pytorch/quantization/_utils.py#L495
        """
        edge_or_node: Tuple[Node, Node]
        if node.target in int8_in_int8_out_ops_pt2e and _is_any_annotated([node]):
            if node.target == torch.ops.aten.max_pool2d.default:
                maxpool_node = node
                if not _is_all_annotated([maxpool_node]):
                    return
                maxpool_node_quantization_annotation = maxpool_node.meta[QUANT_ANNOTATION_KEY] if QUANT_ANNOTATION_KEY in maxpool_node.meta else None
                if maxpool_node_quantization_annotation and maxpool_node_quantization_annotation._is_output_of_quantized_pattern:
                    input_act = maxpool_node.args[0]
                    assert isinstance(input_act, Node)
                    assert isinstance(maxpool_node, Node)
                    edge_or_node = (input_act, maxpool_node)
                    maxpool_node_quantization_annotation.output_qspec = SharedQuantizationSpec(edge_or_node)
            else:
                input_node = node.all_input_nodes[0]
                self._annotate_output_share_observer_as_input(input_node, node)
        return

    def _annotate_linear(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        linear_partitions = get_source_partitions(gm.graph, [torch.nn.Linear, torch.nn.functional.linear])
        linear_partitions = list(itertools.chain(*linear_partitions.values()))
        for partition in linear_partitions:
            if len(partition.output_nodes) > 1:
                raise ValueError('Linear partition cannot have more than one output node')
            linear_node = partition.output_nodes[0]
            if linear_node.op != 'call_function' or linear_node.target not in (torch.ops.aten.linear.default,):
                raise ValueError(f'{linear_node} is not an aten linear operator')
            if _is_annotated([linear_node]):
                continue
            self._annotate_linear_node_helper(linear_node, True, quantization_config)

    def _annotate_linear_unary(self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig) -> None:
        postop_list = [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Tanh]
        fused_partitions: List[tuple] = []
        for postop in postop_list:
            fused_partitions = fused_partitions + find_sequential_partitions(gm, [torch.nn.Linear, postop])
        for fused_partition in fused_partitions:
            linear_partition, unary_partition = fused_partition
            linear_node, unary_node = self._get_output_nodes_of_partitions([linear_partition, unary_partition])
            if linear_node.op != 'call_function' or linear_node.target not in (torch.ops.aten.linear.default,):
                continue
            if _is_annotated([unary_node, linear_node]):
                continue
            self._annotate_linear_node_helper(linear_node, False, quantization_config)
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(_annotated=True, _is_output_of_quantized_pattern=True)

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return cls.supported_config_and_operators