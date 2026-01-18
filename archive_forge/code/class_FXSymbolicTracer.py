from __future__ import annotations
import functools
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.fx
import torch.onnx
import torch.onnx._internal.fx.passes as passes
from torch.onnx._internal import _beartype, exporter, io_adapter
class FXSymbolicTracer(exporter.FXGraphExtractor):
    """Generates a FX GraphModule using torch.fx.symbolic_trace API
    Args:
        concrete_args: Inputs to be partially specialized
            It can be used to remove control flow or data structures.
            For example::
                def f(a, b):
                    if b == True:
                        return a
                    else:
                        return a*2
            FX can typically not trace through this due to the presence of control
            flow. However, we can use `concrete_args` to specialize on the value of
            `b` to trace through this::
                f = fx.symbolic_trace(f, concrete_args={'b': False})
                assert f(3, False)  == 6
            Note that although you can still pass in different values of `b`, they will be ignored.
            It can also be used to eliminate data-structure handling from
            our function. This will use pytrees to flatten your input. To avoid
            overspecializing, pass in `fx.PH` for values that shouldn't be
            specialized. For example::
                def f(x):
                    out = 0
                    for v in x.values():
                        out += v
                    return out
                f = fx.symbolic_trace(f, concrete_args={'x': {'a': fx.PH, 'b': fx.PH, 'c': fx.PH}})
                assert f({'a': 1, 'b': 2, 'c': 4}) == 7
    """

    def __init__(self, concrete_args: Optional[Dict[str, Any]]=None):
        super().__init__()
        self.concrete_args = concrete_args

    @_beartype.beartype
    def _trace_into_fx_graph_via_fx_symbolic_trace(self, model, model_args, model_kwargs) -> torch.fx.GraphModule:
        bind_input_step = io_adapter.BindInputStep(torch.onnx.utils.model_signature(model))
        self.input_adapter.append_step(bind_input_step)
        _, named_args = bind_input_step.apply(model_args, model_kwargs, model=model)
        concrete_args = {}
        for param_name, param_value in named_args.items():
            if isinstance(param_value, torch.Tensor):
                concrete_args[param_name] = torch.fx._symbolic_trace.PH
            else:
                concrete_args[param_name] = param_value
        merge_kwargs_step = io_adapter.MergeKwargsIntoArgsInputStep()
        self.input_adapter.append_step(merge_kwargs_step)
        return _module_expansion_symbolic_trace(model, concrete_args=concrete_args)

    def generate_fx(self, options: exporter.ResolvedExportOptions, model: Union[torch.nn.Module, Callable], model_args: Sequence[Any], model_kwargs: Mapping[str, Any]) -> torch.fx.GraphModule:
        diagnostic_context = options.diagnostic_context
        graph_module = self._trace_into_fx_graph_via_fx_symbolic_trace(model, model_args, model_kwargs)
        graph_module = passes.MovePlaceholderToFront(diagnostic_context, graph_module).run()
        replace_get_attr_with_placeholder_pass = passes.ReplaceGetAttrWithPlaceholder(diagnostic_context, graph_module)
        graph_module = replace_get_attr_with_placeholder_pass.run()
        replaced_attrs = replace_get_attr_with_placeholder_pass.replaced_attrs
        append_extra_input_step = io_adapter.LiftParametersAndBuffersIntoArgsInputStep(replaced_attrs)
        self.input_adapter.append_step(append_extra_input_step)
        graph_module = passes.MovePlaceholderToFront(diagnostic_context, graph_module).run()
        graph_module.recompile()
        updated_model_args = self.input_adapter.apply(*model_args, model=model, **model_kwargs)
        return self.pre_export_passes(options, model, graph_module, updated_model_args)

    @_beartype.beartype
    def pre_export_passes(self, options: exporter.ResolvedExportOptions, original_model: Union[torch.nn.Module, Callable], fx_module: torch.fx.GraphModule, fx_module_args: Sequence[Any]):
        return exporter.common_pre_export_passes(options, original_model, fx_module, fx_module_args)