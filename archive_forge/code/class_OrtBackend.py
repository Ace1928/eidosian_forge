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
@compatibility(is_backward_compatible=False)
class OrtBackend:
    """A backend compiles (sub-)graphs in torch.fx.GraphModule to onnxruntime.InferenceSession calls.

    The compiler entry point is OrtBackend.compile, which
        1. partitions the original graph into supported sub-graphs (type: torch.fx.GraphModule) and unsupported
           sub-graphs.
        2. For each supported sub-graph, it replaces its _wrapped_call function with _ort_accelerated_call.
        3. Inside _ort_accelerated_call, it creates onnxruntime.InferenceSession and calls it to execute the sub-graph.
    """

    def __init__(self, options: Optional[OrtBackendOptions]=None):
        self._options: Final = OrtBackendOptions() if options is None else options
        self._resolved_onnx_exporter_options = torch.onnx._internal.exporter.ResolvedExportOptions(torch.onnx.ExportOptions() if self._options.export_options is None else self._options.export_options)
        support_dict = torch.onnx._internal.fx.decomposition_table._create_onnx_supports_op_overload_table(self._resolved_onnx_exporter_options.onnx_registry)
        extra_support_dict: Dict[str, Any] = {'getattr': None, '_operator.getitem': None}
        self._supported_ops = OrtOperatorSupport(support_dict, extra_support_dict)
        self._partitioner_cache: Dict[torch.fx.GraphModule, torch.fx.GraphModule] = {}
        self._all_ort_execution_info = OrtExecutionInfoForAllGraphModules()
        self._assert_allclose_to_baseline = False
        self.execution_count = 0
        self.run = _run_onnx_session_with_ortvaluevector if hasattr(ORTC, 'push_back_batch') else _run_onnx_session_with_fetch

    def _select_eps(self, graph_module: torch.fx.GraphModule, *args) -> Sequence[Tuple[str, Mapping[str, Any]]]:
        inferred_eps: Tuple[str, ...] = tuple()
        if self._options.infer_execution_providers:
            if (eps_from_args := _infer_ep_from_device(*args)):
                inferred_eps = eps_from_args
            elif (eps_from_graph_module := _infer_ep_from_graph_module(graph_module)):
                inferred_eps = eps_from_graph_module
        selected_eps = []
        for ep in (*(self._options.preferred_execution_providers or []), *_sort_eps(inferred_eps), *(self._options.default_execution_providers or _infer_default_eps())):
            if isinstance(ep, str):
                ep = (ep, {})
            elif isinstance(ep, tuple) and ep[1] is None:
                ep = (ep[0], {})
            if ep is not None and ep not in selected_eps:
                selected_eps.append(ep)
        return selected_eps

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

    def compile(self, graph_module: torch.fx.GraphModule, args) -> torch.fx.GraphModule:
        from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
        if graph_module in self._partitioner_cache:
            partitioned_prim_graph_module = self._partitioner_cache[graph_module]
        else:
            prim_graph_module = graph_module
            _replace_to_copy_with_to(prim_graph_module)
            partitioner = CapabilityBasedPartitioner(prim_graph_module, self._supported_ops, allows_single_node_partition=True)
            partitioned_prim_graph_module = partitioner.partition_and_fuse()
            self._partitioner_cache[graph_module] = partitioned_prim_graph_module
            for node in partitioned_prim_graph_module.graph.nodes:
                if node.op == 'call_module' and 'fused_' in node.name:
                    fused_module = getattr(partitioned_prim_graph_module, node.name)
                    fused_module._wrapped_call = self._ort_acclerated_call
        return partitioned_prim_graph_module

    def __call__(self, graph_module: torch.fx.GraphModule, args) -> torch.fx.GraphModule:
        """If ``OrtBackendOptions.use_aot_autograd`` is ``True``, the `auto_autograd` compiler
        will be invoked, wrapping this ``OrtBackend`` instance's ``compile`` method. Otherwise,
        the ``compile`` method is invoked directly."""
        if self._options.use_aot_autograd:
            from functorch.compile import min_cut_rematerialization_partition
            from torch._dynamo.backends.common import aot_autograd
            return aot_autograd(fw_compiler=self.compile, partition_fn=min_cut_rematerialization_partition, decompositions=self._resolved_onnx_exporter_options.decomposition_table)(graph_module, args)
        return self.compile(graph_module, args)
    __instance_cache_max_count: Final = 8
    __instance_cache: Final[List['OrtBackend']] = []

    @staticmethod
    def get_cached_instance_for_options(options: Optional[Union[OrtBackendOptions, Mapping[str, Any]]]=None) -> 'OrtBackend':
        """Returns a possibly cached instance of an ``OrtBackend``. If an existing
        backend was created previously through this function with the same options,
        it will be returned. Otherwise a new backend will be created, cached, and
        returned.

        Note: if ``options`` sets ``ort_session_options``, a new ``OrtBackend``
        will always be returned, since ``onnxruntime.SessionOptions`` cannot
        participate in caching."""

        def reusable(a: OrtBackendOptions, b: OrtBackendOptions):
            if a.preferred_execution_providers != b.preferred_execution_providers or a.infer_execution_providers != b.infer_execution_providers or a.default_execution_providers != b.default_execution_providers or (a.preallocate_output != b.preallocate_output) or (a.use_aot_autograd != b.use_aot_autograd):
                return False
            if a.ort_session_options is not None or b.ort_session_options is not None:
                return False
            if a.export_options is b.export_options:
                return True
            if a.export_options is not None and b.export_options is not None:
                return a.export_options.dynamic_shapes == b.export_options.dynamic_shapes and a.export_options.op_level_debug == b.export_options.op_level_debug and (a.export_options.diagnostic_options == b.export_options.diagnostic_options) and (a.export_options.onnx_registry is b.export_options.onnx_registry) and (a.export_options.fake_context is b.export_options.fake_context)
            return False
        if not isinstance(options, OrtBackendOptions):
            options = OrtBackendOptions(**options or {})
        backend = next((b for b in OrtBackend.__instance_cache if reusable(b._options, options)), None)
        if backend is None:
            assert len(OrtBackend.__instance_cache) < OrtBackend.__instance_cache_max_count, f'No more than {OrtBackend.__instance_cache_max_count} instances of {OrtBackend} allowed. Please instantiate `{OrtBackend}` explicitly to pass to `torch.compile`. See https://github.com/pytorch/pytorch/pull/107973#discussion_r1306144795 for discussion.'
            OrtBackend.__instance_cache.append((backend := OrtBackend(options)))
        return backend

    @staticmethod
    def clear_cached_instances():
        OrtBackend.__instance_cache.clear()

    @staticmethod
    def get_cached_instances():
        return tuple(OrtBackend.__instance_cache)