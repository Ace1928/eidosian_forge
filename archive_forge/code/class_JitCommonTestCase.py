import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
from torch.testing._internal.common_dtype import floating_and_complex_types_and
from torch.testing._internal.common_utils import TestCase, \
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401
from itertools import chain
from typing import List, Union
from torch._C import TensorType
import io
class JitCommonTestCase(TestCase):

    def createFunctionFromGraph(self, trace):
        graph = trace if isinstance(trace, torch._C.Graph) else trace.graph()
        return torch._C._create_function_from_graph('forward', graph)

    def assertExportImport(self, trace, inputs):
        m = self.createFunctionFromGraph(trace)
        self.assertExportImportModule(m, inputs)

    def assertExportImportModule(self, m, inputs):
        m_import = self.getExportImportCopy(m)
        a = self.runAndSaveRNG(m, inputs)
        b = self.runAndSaveRNG(m_import, inputs)
        self.assertEqual(a, b, 'Results of original model and exported/imported version of model differed')

    def runAndSaveRNG(self, func, inputs, kwargs=None):
        kwargs = kwargs if kwargs else {}
        with freeze_rng_state():
            results = func(*inputs, **kwargs)
        return results

    def getExportImportCopy(self, m, also_test_file=True, map_location=None):
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        imported = torch.jit.load(buffer, map_location=map_location)
        if not also_test_file:
            return imported
        with TemporaryFileName() as fname:
            torch.jit.save(imported, fname)
            return torch.jit.load(fname, map_location=map_location)

    def autoDiffErrorMessage(self, should_autodiff_node, nodes_not_in_diff_graph, fusion_nodes_not_found, non_fusible_nodes_being_fused, fusion_nodes_found, nodes_in_diff_graph):
        err_msg = "\nFailure in testing nodes' autodifferentiation. "
        if should_autodiff_node:
            err_msg += 'One or more nodes were expected to be autodiffed, but were not found in specified fusible/nonfusible DifferentiableGraph groups. \nSpecifically:'
            diff_nodes_missing = []
            diff_nodes_in_fusion = []
            fusion_nodes_missing = []
            fusion_nodes_in_diff = []
            for node in nodes_not_in_diff_graph:
                if node in non_fusible_nodes_being_fused:
                    diff_nodes_in_fusion.append(node)
                else:
                    diff_nodes_missing.append(node)
            for node in fusion_nodes_not_found:
                if node in nodes_in_diff_graph:
                    fusion_nodes_in_diff.append(node)
                else:
                    fusion_nodes_missing.append(node)
            if len(diff_nodes_missing) > 0:
                err_msg += f'\n  {diff_nodes_missing} were not in one of the DifferentiableGraphs when they were expected to be. Did you intend for these nodes to be autodiffed? If not, remove them from the list of nonfusible nodes.'
            if len(diff_nodes_in_fusion) > 0:
                err_msg += f'\n  {diff_nodes_in_fusion} were found in one of the FusionGroups when they were expected to be just in a DifferentiableGraph. If it was intended for these nodes to be in FusionGroups, reclassify these nodes as fusible nodes. If these nodes were not intended to be fused, your autodifferentiation logic might be wrong.'
            if len(fusion_nodes_missing) > 0:
                err_msg += f"\n  {fusion_nodes_missing} were not in one of the FusionGroups of the DifferentiableGraphs when they were expected to be. They were also not found in an outer DifferentiableGraph. Did you intend for these nodes to be autodifferentiated? If not, you should remove these nodes from the test's fusible nodes. Otherwise your autodifferentiation logic might be wrong."
            if len(fusion_nodes_in_diff) > 0:
                err_msg += f"\n  {fusion_nodes_in_diff} were not in one of the FusionGroups of the DifferentiableGraphs when they were expected to be, instead they were found just in an outer DifferentiableGraph. Did you intend for these nodes to be fused? If not, you should move these nodes into the test's nonfusible nodes. Otherwise your autodifferentiation logic might be wrong."
        else:
            err_msg += 'One or more nodes were not expected to be autodiffed but were found in a DifferentiableGraph or in a FusionGroup of a DifferentiableGraph. Did you intend for these nodes to be autodiffed? If so, change this test to expect autodifferentiation. \nSpecifically:'
            if len(fusion_nodes_found) > 0:
                err_msg += f'\n  {fusion_nodes_found} were not expected to be in one of the DifferentiableGraphs, but appeared in a FusionGroup of a DifferentiableGraph. '
            if len(nodes_in_diff_graph) > 0:
                err_msg += f'\n  {nodes_in_diff_graph} were not expected to be in one of the DifferentiableGraphs but were.'
        return err_msg

    def assertAutodiffNode(self, graph, should_autodiff_node, nonfusible_nodes, fusible_nodes):
        diff_nodes = graph.findAllNodes('prim::DifferentiableGraph')
        diff_subgraphs = [node.g('Subgraph') for node in diff_nodes]
        fusion_nodes = list(chain.from_iterable([g.findAllNodes('prim::FusionGroup') for g in diff_subgraphs]))
        fusion_subgraphs = [node.g('Subgraph') for node in fusion_nodes]
        nodes_in_diff_graph = []
        nodes_not_in_diff_graph = []
        non_fusible_nodes_being_fused = []
        for node in nonfusible_nodes:
            if any((g.findNode(node) is not None for g in diff_subgraphs)):
                nodes_in_diff_graph.append(node)
            else:
                nodes_not_in_diff_graph.append(node)
            if any((g.findNode(node) is not None for g in fusion_subgraphs)):
                non_fusible_nodes_being_fused.append(node)
        found_all_nonfusible_nodes = len(nodes_in_diff_graph) == len(nonfusible_nodes)
        fusion_nodes_found = []
        fusion_nodes_not_found = []
        for node in fusible_nodes:
            if any((g.findNode(node) is not None for g in fusion_subgraphs)):
                fusion_nodes_found.append(node)
            else:
                fusion_nodes_not_found.append(node)
        found_all_fusible_nodes = len(fusion_nodes_found) == len(fusible_nodes)
        if should_autodiff_node is not None:
            err_msg = self.autoDiffErrorMessage(should_autodiff_node, nodes_not_in_diff_graph, fusion_nodes_not_found, non_fusible_nodes_being_fused, fusion_nodes_found, nodes_in_diff_graph)
            self.assertEqual(should_autodiff_node, found_all_nonfusible_nodes and found_all_fusible_nodes, err_msg)

    def checkShapeAnalysis(self, out_sizes: Union[List[int], List[List[int]]], traced_graph, assert_propagation, constant_prop=True):
        prev_symbolic_shapes_test_enabled = torch._C._jit_symbolic_shapes_test_mode_enabled()
        for enable_test_mode in [True, False]:
            torch._C._jit_set_symbolic_shapes_test_mode(enable_test_mode)
            torch._C._jit_erase_non_input_shape_information(traced_graph)
            if constant_prop:
                torch._C._jit_pass_constant_propagation(traced_graph)
            torch._C._jit_pass_propagate_shapes_on_graph(traced_graph)
            output = next(traced_graph.outputs()).type()

            def test_type(type, actual_size):
                sizes = type.symbolic_sizes()
                out_type = TensorType.get().with_sizes(sizes)
                actual_type = TensorType.get().with_sizes(actual_size)
                self.assertTrue(actual_type.isSubtypeOf(out_type))
                if assert_propagation:
                    self.assertEqual(out_type.sizes(), actual_size)
            if output.isSubtypeOf(torch._C.TensorType.get()):
                test_type(output, out_sizes)
            else:
                tuple_elements = output.elements()
                for i in range(len(tuple_elements)):
                    test_type(tuple_elements[i], out_sizes[i])
        torch._C._jit_set_symbolic_shapes_test_mode(prev_symbolic_shapes_test_enabled)