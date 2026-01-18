import dataclasses
import importlib
import logging
from typing import (
from typing_extensions import TypeAlias
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx._compatibility import compatibility
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree
def _ort_acclerated_call(self, graph_module: torch.fx.GraphModule, *args, **kwargs):
    """This function replaces GraphModule._wrapped_call in compiled model.

        The _wrapped_call is the underlying implementation of forward method. Replacing
        it means we delegate the computation to _ort_acclerated_call and therefore
        onnxruntime.InferenceSession.
        """
    cached_execution_info_per_session = self._all_ort_execution_info.search_reusable_session_execution_info(graph_module, *args)
    if cached_execution_info_per_session:
        onnx_session = cached_execution_info_per_session.session
        input_names = cached_execution_info_per_session.input_names
        output_names = cached_execution_info_per_session.output_names
        input_devices = cached_execution_info_per_session.input_devices
        output_devices = cached_execution_info_per_session.output_devices
        prim_outputs = cached_execution_info_per_session.example_outputs
    else:
        graph_module = torch.onnx._internal.fx.passes.MovePlaceholderToFront(self._resolved_onnx_exporter_options.diagnostic_context, graph_module).run()
        if self._resolved_onnx_exporter_options.dynamic_shapes:
            self.preallocate_output = False
            extracted_outputs = _extract_graph_module_outputs(graph_module)

            def maybe_map_to_meta_val(value):
                if hasattr(value, 'meta') and 'val' in value.meta:
                    return value.meta['val']
                else:
                    return value
            prim_outputs = _pytree.tree_map(maybe_map_to_meta_val, extracted_outputs)
        else:
            try:
                prim_outputs = FakeTensorProp(graph_module).propagate(*args, **kwargs)
            except Exception:
                logger.warning('FakeTensorProb failed for %s', graph_module)
                self.preallocate_output = False
                raise
        fx_interpreter = fx_onnx_interpreter.FxOnnxInterpreter(diagnostic_context=self._resolved_onnx_exporter_options.diagnostic_context)
        graph_module = torch.onnx._internal.fx.passes.InsertTypePromotion(self._resolved_onnx_exporter_options.diagnostic_context, graph_module).run()
        exported = fx_interpreter.run(fx_graph_module=graph_module, onnxfunction_dispatcher=self._resolved_onnx_exporter_options.onnxfunction_dispatcher, op_level_debug=self._resolved_onnx_exporter_options.op_level_debug)
        onnx_model = exported.to_model_proto(opset_version=self._resolved_onnx_exporter_options.onnx_registry.opset_version)
        onnx_session = onnxruntime.InferenceSession(path_or_bytes=onnx_model.SerializeToString(), sess_options=self._options.ort_session_options, providers=self._select_eps(graph_module, *args))
        input_names = tuple((input.name for input in onnx_model.graph.input))
        output_names = tuple((output.name for output in onnx_model.graph.output))
        input_devices = _get_onnx_devices(args)
        if isinstance(prim_outputs, tuple):
            output_devices = _get_onnx_devices(prim_outputs)
        else:
            output_devices = _get_onnx_devices((prim_outputs,))
        execution_info_per_session = OrtExecutionInfoPerSession(session=onnx_session, input_names=input_names, input_value_infos=tuple((input for input in onnx_model.graph.input)), output_names=output_names, output_value_infos=tuple((output for output in onnx_model.graph.output)), input_devices=input_devices, output_devices=output_devices, example_outputs=prim_outputs)
        self._all_ort_execution_info.cache_session_execution_info(graph_module, execution_info_per_session)
    self.execution_count += 1
    is_single_tensor_output = isinstance(prim_outputs, torch.Tensor)
    normalized_prim_outputs = (prim_outputs,) if is_single_tensor_output else prim_outputs
    assert isinstance(normalized_prim_outputs, tuple)
    assert all((isinstance(elem, torch.Tensor) for elem in normalized_prim_outputs))
    _nvtx_range_push('run_onnx_session_with_ortvaluevector')
    onnx_outputs = self.run(onnx_session, input_names, args, input_devices, output_names, normalized_prim_outputs, output_devices, self._options.preallocate_output)
    _nvtx_range_pop()
    if self._assert_allclose_to_baseline:
        baseline_outputs = torch._prims.executor.execute(graph_module, *args, executor='aten')
        normalized_baseline_ouptuts = (baseline_outputs,) if is_single_tensor_output else baseline_outputs
        for onnx_output, baseline_output in zip(onnx_outputs, normalized_baseline_ouptuts):
            torch.testing.assert_close(onnx_output, baseline_output)
    return onnx_outputs[0] if is_single_tensor_output else onnx_outputs