import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.nn.intrinsic import _FusedModule
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM
from torch.ao.quantization import (
from torch.ao.quantization import QuantWrapper, QuantStub, DeQuantStub, \
from torch.ao.quantization.quantization_mappings import (
from torch.testing._internal.common_quantized import (
from torch.jit.mobile import _load_for_lite_interpreter
import copy
import io
import functools
import time
import os
import unittest
import numpy as np
from torch.testing import FileCheck
from typing import Callable, Tuple, Dict, Any, Union, Type, Optional
import torch._dynamo as torchdynamo
class QuantizationTestCase(TestCase):

    def setUp(self):
        super().setUp()
        self.calib_data = [[torch.rand(2, 5, dtype=torch.float)] for _ in range(2)]
        self.train_data = [[torch.rand(2, 5, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long)] for _ in range(2)]
        self.img_data_1d = [[torch.rand(2, 3, 10, dtype=torch.float)] for _ in range(2)]
        self.img_data_2d = [[torch.rand(1, 3, 10, 10, dtype=torch.float)] for _ in range(2)]
        self.img_data_3d = [[torch.rand(1, 3, 5, 5, 5, dtype=torch.float)] for _ in range(2)]
        self.img_data_1d_train = [[torch.rand(2, 3, 10, dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)] for _ in range(2)]
        self.img_data_2d_train = [[torch.rand(1, 3, 10, 10, dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)] for _ in range(2)]
        self.img_data_3d_train = [[torch.rand(1, 3, 5, 5, 5, dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)] for _ in range(2)]
        self.img_data_dict = {1: self.img_data_1d, 2: self.img_data_2d, 3: self.img_data_3d}
        self.static_quant_types = [QuantType.STATIC, QuantType.QAT]
        self.all_quant_types = [QuantType.DYNAMIC, QuantType.STATIC, QuantType.QAT]

    def checkNoPrepModules(self, module):
        """Checks the module does not contain child
            modules for quantization preparation, e.g.
            quant, dequant and observer
        """
        self.assertFalse(hasattr(module, 'quant'))
        self.assertFalse(hasattr(module, 'dequant'))

    def checkNoQconfig(self, module):
        """Checks the module does not contain qconfig
        """
        self.assertFalse(hasattr(module, 'qconfig'))
        for child in module.children():
            self.checkNoQconfig(child)

    def checkHasPrepModules(self, module):
        """Checks the module contains child
            modules for quantization preparation, e.g.
            quant, dequant and observer
        """
        self.assertTrue(hasattr(module, 'module'))
        self.assertTrue(hasattr(module, 'quant'))
        self.assertTrue(hasattr(module, 'dequant'))

    def checkObservers(self, module, propagate_qconfig_list=None, prepare_custom_config_dict=None):
        """Checks the module or module's leaf descendants
            have observers in preparation for quantization
        """
        if propagate_qconfig_list is None:
            propagate_qconfig_list = get_default_qconfig_propagation_list()
        if prepare_custom_config_dict is None:
            prepare_custom_config_dict = {}
        float_to_observed_module_class_mapping = prepare_custom_config_dict.get('float_to_observed_custom_module_class', {})

        def is_leaf_module(module):
            submodule_name_count = 0
            for name, _ in module.named_children():
                if name != 'activation_post_process':
                    submodule_name_count += 1
            return submodule_name_count == 0
        if hasattr(module, 'qconfig') and module.qconfig is not None and (is_leaf_module(module) and (not isinstance(module, torch.nn.Sequential)) and (type(module) in propagate_qconfig_list) or type(module) in float_to_observed_module_class_mapping.keys()) and (not isinstance(module, torch.ao.quantization.DeQuantStub)):
            self.assertTrue(hasattr(module, 'activation_post_process'), 'module: ' + str(type(module)) + ' do not have observer')
        if type(module) not in get_default_qat_module_mappings().values() and type(module) not in float_to_observed_module_class_mapping.values() and (not isinstance(module, _FusedModule)):
            for child in module.children():
                if type(child) in [nn.Dropout]:
                    continue
                self.checkObservers(child, propagate_qconfig_list, prepare_custom_config_dict)

    def checkQuantDequant(self, mod):
        """Checks that mod has nn.Quantize and
            nn.DeQuantize submodules inserted
        """
        self.assertEqual(type(mod.quant), nnq.Quantize)
        self.assertEqual(type(mod.dequant), nnq.DeQuantize)

    def checkWrappedQuantizedLinear(self, mod):
        """Checks that mod has been swapped for an nnq.Linear
            module, the bias is qint32, and that the module
            has Quantize and DeQuantize submodules
        """
        self.assertEqual(type(mod.module), nnq.Linear)
        self.checkQuantDequant(mod)

    def checkQuantizedLinear(self, mod):
        self.assertEqual(type(mod), nnq.Linear)

    def checkDynamicQuantizedLinear(self, mod, dtype):
        """Checks that mod has been swapped for an nnqd.Linear
            module, the bias is float.
        """
        self.assertEqual(type(mod), nnqd.Linear)
        self.assertEqual(mod._packed_params.dtype, dtype)

    def checkDynamicQuantizedLinearRelu(self, mod, dtype):
        """Checks that mod has been swapped for an nnqd.Linear
            module, the bias is float.
        """
        self.assertEqual(type(mod), nniqd.LinearReLU)
        self.assertEqual(mod._packed_params.dtype, dtype)

    def check_eager_serialization(self, ref_model, loaded_model, x):
        model_dict = ref_model.state_dict()
        b = io.BytesIO()
        torch.save(model_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        loaded_model.load_state_dict(loaded_dict)
        ref_out = ref_model(*x)
        load_out = loaded_model(*x)

        def check_outputs(ref_out, load_out):
            self.assertEqual(ref_out[0], load_out[0])
            if isinstance(ref_out[1], tuple):
                self.assertEqual(ref_out[1][0], load_out[1][0])
                self.assertEqual(ref_out[1][1], load_out[1][1])
            else:
                self.assertEqual(ref_out[1], load_out[1])
        check_outputs(ref_out, load_out)
        b = io.BytesIO()
        torch.save(ref_model, b)
        b.seek(0)
        loaded = torch.load(b)
        load_out = loaded(*x)
        check_outputs(ref_out, load_out)

    def check_weight_bias_api(self, ref_model, weight_keys, bias_keys):
        weight = ref_model.get_weight()
        bias = ref_model.get_bias()
        self.assertEqual(weight_keys ^ weight.keys(), set())
        self.assertEqual(bias_keys ^ bias.keys(), set())

    def checkDynamicQuantizedLSTM(self, mod, reference_module_type, dtype):
        """Checks that mod has been swapped for an nnqd.LSTM type
            module, the bias is float.
        """
        wt_dtype_map = {torch.qint8: 'quantized_dynamic', torch.float16: 'quantized_fp16'}
        self.assertEqual(type(mod), reference_module_type)
        for packed_params in mod._all_weight_values:
            self.assertEqual(packed_params.param.__getstate__()[0][0], wt_dtype_map[dtype])

    def checkLinear(self, mod):
        self.assertEqual(type(mod), torch.nn.Linear)

    def checkDynamicQuantizedModule(self, mod, reference_module_type, dtype):
        """Checks that mod has been swapped for an nnqd.Linear
            module, the bias is float.
        """
        wt_dtype_map = {torch.qint8: 'quantized_dynamic', torch.float16: 'quantized_fp16'}
        self.assertEqual(type(mod), reference_module_type)
        if hasattr(mod, '_all_weight_values'):
            for packed_params in mod._all_weight_values:
                self.assertEqual(packed_params.param.__getstate__()[0][0], wt_dtype_map[dtype])

    def checkScriptable(self, orig_mod, calib_data, check_save_load=False):
        scripted = torch.jit.script(orig_mod)
        self._checkScriptable(orig_mod, scripted, calib_data, check_save_load)
        traced = torch.jit.trace(orig_mod, calib_data[0])
        self._checkScriptable(orig_mod, traced, calib_data, check_save_load)

    def _checkScriptable(self, orig_mod, script_mod, calib_data, check_save_load):
        self._checkModuleCorrectnessAgainstOrig(orig_mod, script_mod, calib_data)
        buffer = io.BytesIO()
        torch.jit.save(script_mod, buffer)
        buffer.seek(0)
        loaded_mod = torch.jit.load(buffer)
        if check_save_load:
            self._checkModuleCorrectnessAgainstOrig(orig_mod, loaded_mod, calib_data)

    def _checkModuleCorrectnessAgainstOrig(self, orig_mod, test_mod, calib_data):
        for inp in calib_data:
            ref_output = orig_mod(*inp)
            scripted_output = test_mod(*inp)
            self.assertEqual(scripted_output, ref_output)

    def checkGraphModeOp(self, module, inputs, quantized_op, tracing=False, debug=False, check=True, eval_mode=True, dynamic=False, qconfig=None):
        if debug:
            print('Testing:', str(module))
        qconfig_dict = {'': get_default_qconfig(torch.backends.quantized.engine)}
        if eval_mode:
            module = module.eval()
        if dynamic:
            qconfig_dict = {'': default_dynamic_qconfig if qconfig is None else qconfig}
        model = get_script_module(module, tracing, inputs[0]).eval()
        if debug:
            print('input graph:', model.graph)
        models = {}
        outputs = {}
        for debug in [True, False]:
            if dynamic:
                models[debug] = quantize_dynamic_jit(model, qconfig_dict, debug=debug)
                outputs[debug] = models[debug](inputs)
            else:
                inputs_copy = copy.deepcopy(inputs)
                models[debug] = quantize_jit(model, qconfig_dict, test_only_eval_fn, [inputs_copy], inplace=False, debug=debug)
                outputs[debug] = models[debug](*inputs[0])
        if debug:
            print('debug graph:', models[True].graph)
            print('non debug graph:', models[False].graph)
        if check:
            self.assertEqual(outputs[True], outputs[False])
            FileCheck().check(quantized_op).run(models[False].graph)
        return models[False]

    def checkGraphModuleNodes(self, graph_module, expected_node=None, expected_node_occurrence=None, expected_node_list=None):
        """ Check if GraphModule contains the target node
        Args:
            graph_module: the GraphModule instance we want to check
            expected_node, expected_node_occurrence, expected_node_list:
               see docs for checkGraphModeFxOp
        """
        nodes_in_graph = {}
        node_list = []
        modules = dict(graph_module.named_modules(remove_duplicate=False))
        for node in graph_module.graph.nodes:
            n = None
            if node.op == 'call_function' or node.op == 'call_method':
                n = NodeSpec(node.op, node.target)
            elif node.op == 'call_module':
                n = NodeSpec(node.op, type(modules[node.target]))
            if n is not None:
                node_list.append(n)
                if n in nodes_in_graph:
                    nodes_in_graph[n] += 1
                else:
                    nodes_in_graph[n] = 1
        if expected_node is not None:
            self.assertTrue(expected_node in nodes_in_graph, 'node:' + str(expected_node) + ' not found in the graph module')
        if expected_node_occurrence is not None:
            for expected_node, occurrence in expected_node_occurrence.items():
                if occurrence != 0:
                    self.assertTrue(expected_node in nodes_in_graph, 'Check failed for node:' + str(expected_node) + ' not found')
                    self.assertTrue(nodes_in_graph[expected_node] == occurrence, 'Check failed for node:' + str(expected_node) + ' Expected occurrence:' + str(occurrence) + ' Found occurrence:' + str(nodes_in_graph[expected_node]))
                else:
                    self.assertTrue(expected_node not in nodes_in_graph, 'Check failed for node:' + str(expected_node) + ' expected no occurrence but found')
        if expected_node_list is not None:
            cur_index = 0
            for n in node_list:
                if cur_index == len(expected_node_list):
                    return
                if n == expected_node_list[cur_index]:
                    cur_index += 1
            self.assertTrue(cur_index == len(expected_node_list), 'Check failed for graph:' + self.printGraphModule(graph_module, print_str=False) + 'Expected ordered list:' + str(expected_node_list))

    def printGraphModule(self, graph_module, print_str=True):
        modules = dict(graph_module.named_modules(remove_duplicate=False))
        node_infos = []
        for n in graph_module.graph.nodes:
            node_info = ' '.join(map(repr, [n.op, n.name, n.target, n.args, n.kwargs]))
            if n.op == 'call_module':
                node_info += ' module type: ' + repr(type(modules[n.target]))
            node_infos.append(node_info)
        str_to_print = '\n'.join(node_infos)
        if print_str:
            print(str_to_print)
        return str_to_print
    if HAS_FX:

        def assert_types_for_matched_subgraph_pairs(self, matched_subgraph_pairs: Dict[str, Tuple[NSSubgraph, NSSubgraph]], expected_types: Dict[str, Tuple[Tuple[Callable, Callable], Tuple[Callable, Callable]]], gm_a: GraphModule, gm_b: GraphModule) -> None:
            """
            Verifies that the types specified in expected_types match
            the underlying objects pointed to by the nodes in matched_subgraph_pairs.

            An example successful test case:

              matched_subgraph_pairs = {'x0': (graph_a_conv_0_node, graph_b_conv_0_node)}
              expected_types = {'x0': (nn.Conv2d, nnq.Conv2d)}

            The function tests for key equivalence, and verifies types with
            instance checks.
            """

            def _get_underlying_op_type(node: Node, gm: GraphModule) -> Union[Callable, str]:
                if node.op == 'call_module':
                    mod = getattr(gm, node.target)
                    return type(mod)
                else:
                    assert node.op in ('call_function', 'call_method')
                    return node.target
            self.assertTrue(len(matched_subgraph_pairs) == len(expected_types), f'Expected length of results to match, but got {len(matched_subgraph_pairs)} and {len(expected_types)}')
            for k, v in expected_types.items():
                expected_types_a, expected_types_b = v
                exp_type_start_a, exp_type_end_a = expected_types_a
                exp_type_start_b, exp_type_end_b = expected_types_b
                subgraph_a, subgraph_b = matched_subgraph_pairs[k]
                act_type_start_a = _get_underlying_op_type(subgraph_a.start_node, gm_a)
                act_type_start_b = _get_underlying_op_type(subgraph_b.start_node, gm_b)
                act_type_end_a = _get_underlying_op_type(subgraph_a.end_node, gm_a)
                act_type_end_b = _get_underlying_op_type(subgraph_b.end_node, gm_b)
                types_match = exp_type_start_a is act_type_start_a and exp_type_end_a is act_type_end_a and (exp_type_start_b is act_type_start_b) and (exp_type_end_b is act_type_end_b)
                self.assertTrue(types_match, 'Type mismatch at {}: expected {}, got {}'.format(k, (exp_type_start_a, exp_type_end_a, exp_type_start_b, exp_type_end_b), (act_type_start_a, act_type_end_a, act_type_start_b, act_type_end_b)))

        def assert_ns_compare_dict_valid(self, act_compare_dict: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
            """
            Verifies that the act_compare_dict (output of Numeric Suite APIs) is valid:
            1. for each layer, results are recorded for two models
            2. number of seen tensors match
            3. shapes of each pair of seen tensors match
            """
            for layer_name, result_type_to_data in act_compare_dict.items():
                for result_type, layer_data in result_type_to_data.items():
                    self.assertTrue(len(layer_data) == 2, f'Layer {layer_name} does not have exactly two model results.')
                    model_name_0, model_name_1 = layer_data.keys()
                    for res_idx in range(len(layer_data[model_name_0])):
                        layer_data_0 = layer_data[model_name_0][res_idx]
                        layer_data_1 = layer_data[model_name_1][res_idx]
                        self.assertTrue(layer_data_0['type'] == layer_data_0['type'], f'Layer {layer_name}, {model_name_0} and {model_name_1} do not have the same type.')
                        self.assertTrue(len(layer_data_0['values']) == len(layer_data_1['values']), f'Layer {layer_name}, {model_name_0} and {model_name_1} do not have the same number of seen Tensors.')
                        is_weight_functional_conv1d = result_type == NSSingleResultValuesType.WEIGHT.value and ('conv1d' in layer_data_0['prev_node_target_type'] or 'conv1d' in layer_data_1['prev_node_target_type'])
                        if not is_weight_functional_conv1d:
                            for idx in range(len(layer_data_0['values'])):
                                values_0 = layer_data_0['values'][idx]
                                values_1 = layer_data_1['values'][idx]
                                if isinstance(values_0, torch.Tensor):
                                    self.assertTrue(values_0.shape == values_1.shape, f'Layer {layer_name}, {model_name_0} and {model_name_1} ' + f'have a shape mismatch at idx {idx}.')
                                elif isinstance(values_0, list):
                                    values_0 = values_0[0]
                                    values_1 = values_1[0]
                                    self.assertTrue(values_0.shape == values_1.shape, f'Layer {layer_name}, {model_name_0} and {model_name_1} ' + f'have a shape mismatch at idx {idx}.')
                                else:
                                    assert isinstance(values_0, tuple), f'unhandled type {type(values_0)}'
                                    assert len(values_0) == 2
                                    assert len(values_0[1]) == 2
                                    assert values_0[0].shape == values_1[0].shape
                                    assert values_0[1][0].shape == values_1[1][0].shape
                                    assert values_0[1][1].shape == values_1[1][1].shape
                        ref_node_name_0 = layer_data_0['ref_node_name']
                        ref_node_name_1 = layer_data_1['ref_node_name']
                        prev_node_name_0 = layer_data_0['prev_node_name']
                        prev_node_name_1 = layer_data_1['prev_node_name']
                        if layer_data_0['type'] == NSSingleResultValuesType.NODE_OUTPUT.value:
                            self.assertTrue(ref_node_name_0 == prev_node_name_0)
                            self.assertTrue(ref_node_name_1 == prev_node_name_1)
                        elif layer_data_0['type'] == NSSingleResultValuesType.NODE_INPUT.value:
                            self.assertTrue(ref_node_name_0 != prev_node_name_0)
                            self.assertTrue(ref_node_name_1 != prev_node_name_1)

        def checkGraphModeFxOp(self, model, inputs, quant_type, expected_node=None, expected_node_occurrence=None, expected_node_list=None, is_reference=False, print_debug_info=False, custom_qconfig_dict=None, prepare_expected_node=None, prepare_expected_node_occurrence=None, prepare_expected_node_list=None, prepare_custom_config=None, backend_config=None):
            """ Quantizes model with graph mode quantization on fx and check if the
                quantized model contains the quantized_node

                Args:
                    model: floating point torch.nn.Module
                    inputs: one positional sample input arguments for model
                    expected_node: NodeSpec
                        e.g. NodeSpec.call_function(torch.quantize_per_tensor)
                    expected_node_occurrence: a dict from NodeSpec to
                        expected number of occurrences (int)
                        e.g. {NodeSpec.call_function(torch.quantize_per_tensor) : 1,
                                NodeSpec.call_method('dequantize'): 1}
                    expected_node_list: a list of NodeSpec, used to check the order
                        of the occurrence of Node
                        e.g. [NodeSpec.call_function(torch.quantize_per_tensor),
                                NodeSpec.call_module(nnq.Conv2d),
                                NodeSpec.call_function(F.hardtanh_),
                                NodeSpec.call_method('dequantize')]
                    is_reference: if True, enables reference mode
                    print_debug_info: if True, prints debug info
                    custom_qconfig_dict: overrides default qconfig_dict
                    prepare_expected_node: same as expected_node, but for prepare
                    prepare_expected_node_occurrence: same as
                        expected_node_occurrence, but for prepare
                    prepare_expected_node_list: same as expected_node_list, but
                        for prepare

                Returns:
                    A dictionary with the following structure:
                   {
                       "prepared": ...,  # the prepared model
                       "quantized": ...,  # the quantized non-reference model
                       "quantized_reference": ...,  # the quantized reference model
                       "result": ...,  # the result for either quantized or
                                       # quantized_reference model depending on the
                                       # is_reference argument
                   }
            """
            if type(inputs) == list:
                inputs = inputs[0]
            if quant_type == QuantType.QAT:
                qconfig_mapping = get_default_qat_qconfig_mapping(torch.backends.quantized.engine)
                model.train()
            elif quant_type == QuantType.STATIC:
                qconfig_mapping = get_default_qconfig_mapping(torch.backends.quantized.engine)
                model.eval()
            else:
                qconfig = default_dynamic_qconfig
                qconfig_mapping = QConfigMapping().set_global(qconfig)
                model.eval()
            if quant_type == QuantType.QAT:
                prepare = prepare_qat_fx
            else:
                prepare = prepare_fx
            if custom_qconfig_dict is not None:
                assert type(custom_qconfig_dict) in (QConfigMapping, dict), 'custom_qconfig_dict should be a QConfigMapping or a dict'
                if isinstance(custom_qconfig_dict, QConfigMapping):
                    qconfig_mapping = custom_qconfig_dict
                else:
                    qconfig_mapping = QConfigMapping.from_dict(custom_qconfig_dict)
            prepared = prepare(model, qconfig_mapping, example_inputs=inputs, prepare_custom_config=prepare_custom_config, backend_config=backend_config)
            if not quant_type == QuantType.DYNAMIC:
                prepared(*inputs)
            if print_debug_info:
                print()
                print('quant type:\n', quant_type)
                print('original model:\n', model)
                print()
                print('prepared model:\n', prepared)
            self.checkGraphModuleNodes(prepared, prepare_expected_node, prepare_expected_node_occurrence, prepare_expected_node_list)
            prepared_copy = copy.deepcopy(prepared)
            qgraph = convert_fx(copy.deepcopy(prepared))
            qgraph_reference = convert_to_reference_fx(copy.deepcopy(prepared))
            result = qgraph(*inputs)
            result_reference = qgraph_reference(*inputs)
            qgraph_copy = copy.deepcopy(qgraph)
            qgraph_reference_copy = copy.deepcopy(qgraph_reference)
            qgraph_to_check = qgraph_reference if is_reference else qgraph
            if print_debug_info:
                print()
                print('quantized model:\n', qgraph_to_check)
                self.printGraphModule(qgraph_to_check)
                print()
            self.checkGraphModuleNodes(qgraph_to_check, expected_node, expected_node_occurrence, expected_node_list)
            return {'prepared': prepared_copy, 'quantized': qgraph_copy, 'quantized_reference': qgraph_reference_copy, 'quantized_output': result, 'quantized_reference_output': result_reference}

    def checkEmbeddingSerialization(self, qemb, num_embeddings, embedding_dim, indices, offsets, set_qconfig, is_emb_bag, dtype=torch.quint8):
        if is_emb_bag:
            inputs = [indices, offsets]
        else:
            inputs = [indices]
        emb_dict = qemb.state_dict()
        b = io.BytesIO()
        torch.save(emb_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        embedding_unpack = torch.ops.quantized.embedding_bag_unpack
        for key in emb_dict:
            if isinstance(emb_dict[key], torch._C.ScriptObject):
                assert isinstance(loaded_dict[key], torch._C.ScriptObject)
                emb_weight = embedding_unpack(emb_dict[key])
                loaded_weight = embedding_unpack(loaded_dict[key])
                self.assertEqual(emb_weight, loaded_weight)
        if is_emb_bag:
            loaded_qemb = nnq.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embedding_dim, include_last_offset=True, mode='sum', dtype=dtype)
        else:
            loaded_qemb = nnq.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, dtype=dtype)
        self.check_eager_serialization(qemb, loaded_qemb, inputs)
        loaded_qemb.load_state_dict(loaded_dict)
        self.assertEqual(embedding_unpack(qemb._packed_params._packed_weight), embedding_unpack(loaded_qemb._packed_params._packed_weight))
        self.checkScriptable(qemb, [inputs], check_save_load=True)
        if is_emb_bag:
            float_embedding = torch.nn.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embedding_dim, include_last_offset=True, scale_grad_by_freq=False, mode='sum')
        else:
            float_embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        if set_qconfig:
            float_qparams_observer = PerChannelMinMaxObserver.with_args(dtype=dtype, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0)
            float_embedding.qconfig = QConfig(activation=default_dynamic_quant_observer, weight=float_qparams_observer)
        prepare_dynamic(float_embedding)
        float_embedding(*inputs)
        if is_emb_bag:
            q_embeddingbag = nnq.EmbeddingBag.from_float(float_embedding)
            expected_name = 'QuantizedEmbeddingBag'
        else:
            q_embeddingbag = nnq.Embedding.from_float(float_embedding)
            expected_name = 'QuantizedEmbedding'
        q_embeddingbag(*inputs)
        self.assertTrue(expected_name in str(q_embeddingbag))