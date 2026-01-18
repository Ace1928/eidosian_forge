from __future__ import annotations
import logging
import operator
import types
from typing import (
import torch
import torch._ops
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
class OnnxFunctionDispatcher:
    """A dispatcher that finds the best ONNX Function for ATen/Custom operators.

    It uses the `torch.ops` name to find the function. If not found, it falls back to default.
    Otherwise, the best match is found among all function overloads. An exact match has
    higher precedence over the closest ones.

    Below is a breakdown on how the dispatch mechanism works:

    1. Use the torch.ops name to find the function:
        a. Check if the ATen overload exists in the registry.
        b. If not, check if the default overload exists in the registry.

    2. Find the nearest match among all overloaded functions:
        a. If the types match perfectly, select the function.
        b. Otherwise, find the nearest one with the highest matching score. Because of
            the potential wrongly annotated dtypes and attributes matching, we use
            nearest match to find the best function once the aten name is targeted.

    3. Tie-breaker: If there are multiple nearest matches, we will select the one with
        the highest matching score.

    NOTE: The nearest match `doesn't guarantee` a correct match, and a warning message is logged.
    """

    def __init__(self, onnx_registry: 'OnnxRegistry', diagnostic_context: diagnostics.DiagnosticContext):
        """Initialize the ONNX Function dispatcher.

        Args:
            onnx_registry: The ONNX registry.
            diagnostic_context: The diagnostic context to use for reporting errors.
        """
        self.onnx_registry = onnx_registry
        self.diagnostic_context = diagnostic_context

    @_beartype.beartype
    def dispatch(self, node: torch.fx.Node, onnx_args: Sequence[Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]], onnx_kwargs: Dict[str, fx_type_utils.Argument], diagnostic_context: diagnostics.DiagnosticContext) -> Union['onnxscript.OnnxFunction', 'onnxscript.TracedOnnxFunction']:
        """Dispatches an ONNX function based on the given FX node, arguments, and keyword arguments.
        Args:
            node: The TorchFX node to dispatch the function for.
            onnx_args: The arguments of the ONNX function.
            onnx_kwargs: The keyword arguments of the ONNX function.
            diagnostic_context: The diagnostic context to use for reporting errors.
        Returns:
            Either an `onnxscript.OnnxFunction` or `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm.
        Raises:
            RuntimeError: If there are no overloaded functions available for the given FX node.
        """
        default_and_custom_functions = self.get_function_overloads(node, diagnostic_context)
        return self._find_the_perfect_or_nearest_match_onnxfunction(node, default_and_custom_functions, onnx_args, onnx_kwargs, diagnostic_context)

    @_beartype.beartype
    def _filter_or_keep_complex(self, node, default_and_custom_functions: List[registration.ONNXFunction], diagnostic_context: diagnostics.DiagnosticContext) -> List[registration.ONNXFunction]:
        if any((torch.is_complex(arg.meta['val']) for arg in node.args if isinstance(arg, torch.fx.Node) and 'val' in arg.meta and isinstance(arg.meta['val'], torch.Tensor))):
            default_and_custom_functions = [func for func in default_and_custom_functions if func.is_complex]
            if not default_and_custom_functions:
                op_full_name = self._get_aten_name(node, diagnostic_context).qualified_name()
                diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Cannot find any COMPLEX symbolic function for {op_full_name}, which should be registered under {node.target}.', unsupported_fx_node=node)
                diagnostic_context.log(diagnostic)
                raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
        else:
            default_and_custom_functions = [func for func in default_and_custom_functions if not func.is_complex]
            if not default_and_custom_functions:
                op_full_name = self._get_aten_name(node, diagnostic_context).qualified_name()
                diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Can ONLY find COMPLEX symbolic function for {op_full_name}, which should be registered under {node.target}.', unsupported_fx_node=node)
                diagnostic_context.log(diagnostic)
                raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
        return default_and_custom_functions

    @_beartype.beartype
    @diagnostics.diagnose_call(diagnostics.rules.find_opschema_matched_symbolic_function, diagnostic_message_formatter=_find_opschema_matched_symbolic_function_disagnostic_message_formatter)
    def _find_the_perfect_or_nearest_match_onnxfunction(self, node: torch.fx.Node, default_and_custom_functions: List[registration.ONNXFunction], onnx_args: Sequence[Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]], onnx_kwargs: Dict[str, fx_type_utils.Argument], diagnostic_context: diagnostics.DiagnosticContext):
        """Find the perfect/nearest matched OnnxFunction for the given FX node, arguments, and keyword arguments.

        Args:
            default_and_custom_functions: The list includes overloaded functions, with
                custom ones appearing after the default ones.
            onnx_args: Arguments organized in PyTorch inputs way.
            onnx_kwargs: Keyword arguments organized in PyTorch inputs way.
            diagnostic_context: The diagnostic context to use for reporting errors.

            Returns:
                Either an `onnxscript.OnnxFunction` or `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm.
            Raises:
                RuntimeError: If there are no overloaded functions available for the given FX node.
        """
        overload_match_ranking: Dict[registration.ONNXFunction, Optional[int]] = {}
        diagnostic = diagnostic_context.inflight_diagnostic()
        for symbolic_function in reversed(default_and_custom_functions):
            function_opschema = _OnnxSchemaChecker(symbolic_function.onnx_function)
            if function_opschema.perfect_match_inputs(diagnostic, onnx_args, onnx_kwargs):
                return symbolic_function.onnx_function
            overload_match_ranking[symbolic_function] = function_opschema.match_score
        overload_match_ranking = {k: v for k, v in overload_match_ranking.items() if v is not None}
        if not overload_match_ranking:
            op_full_name = self._get_aten_name(node, diagnostic_context).qualified_name()
            diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Cannot find any perfect/nearest match of symbolic function for {op_full_name},which should be registered under {node.target}.', unsupported_fx_node=node)
            diagnostic_context.log(diagnostic)
            raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
        diagnostic.warning('### Exact match is not found!\nCannot find a perfect match of symbolic overload, a nearest match is found. Please check the ONNX output carefully. \n')
        diagnostic.level = diagnostics.levels.WARNING
        symbolic_function_list: List[registration.ONNXFunction] = sorted(overload_match_ranking, key=lambda k: (overload_match_ranking[k], k.is_custom, default_and_custom_functions.index(k)), reverse=True)
        return symbolic_function_list[0].onnx_function

    @_beartype.beartype
    def _get_aten_name(self, node: torch.fx.Node, diagnostic_context: diagnostics.DiagnosticContext) -> registration.OpName:
        """Get the OpName from the target.

        Args:
            node: The TorchFX node to get the aten name for.
            diagnostic_context: The diagnostic context to use for reporting errors.

        Returns:
            The internal op name within dataclass: registration.OpName.
        """
        if node.target == operator.getitem:
            return registration.OpName.from_name_parts(namespace='aten', op_name='getitem')
        if isinstance(node.target, torch._ops.OpOverloadPacket):
            if node.target != torch.ops.aten.sym_size:
                diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Unsupported OverloadPacket: {node.target}, aten.sym_size is the only allowed OverloadPacket!', unsupported_fx_node=node)
                diagnostic_context.log(diagnostic)
                raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
            aten_op_default = node.target.default
            return registration.OpName.from_op_overload(op_overload=aten_op_default)
        if isinstance(node.target, types.BuiltinFunctionType):
            for node_arg in node.args:
                if not isinstance(node_arg, (torch.fx.Node, int, float)) or (isinstance(node_arg, torch.fx.Node) and (not fx_type_utils.is_torch_symbolic_type(node_arg.meta['val']))):
                    diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Unsupported node arg: {node_arg} (type {type(node_arg)}) with builtin function: {node.target}, only int/float/SymInt/SymFloat is supported with built-in ops!', unsupported_fx_node=node)
                    diagnostic_context.log(diagnostic)
                    raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
            return registration.OpName.from_builtin_function(node.target)
        if isinstance(node.target, torch._ops.OpOverload):
            return registration.OpName.from_op_overload(op_overload=node.target)
        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Unknown call_function target: {node.target}', unsupported_fx_node=node)
        diagnostic_context.log(diagnostic)
        raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)

    @_beartype.beartype
    @diagnostics.diagnose_call(diagnostics.rules.find_operator_overloads_in_onnx_registry, diagnostic_message_formatter=_find_operator_overloads_in_onnx_registry_disagnostic_message_formatter)
    def get_function_overloads(self, node: torch.fx.Node, diagnostic_context: diagnostics.DiagnosticContext) -> List[registration.ONNXFunction]:
        """Get the function overloads from the registry.

        Args:
            node: The node to get the function overloads for.
            diagnostic_context: The diagnostic context to use for reporting errors.

        Returns:
            The list contains ONNXFunctions, starting with the default ones and
            followed by any custom ones.
        """
        internal_opname: registration.OpName = self._get_aten_name(node=node, diagnostic_context=diagnostic_context)
        function_group: Optional[List[registration.ONNXFunction]] = None
        function_group = self.onnx_registry.get_op_functions(namespace=internal_opname.namespace, op_name=internal_opname.op_name, overload=internal_opname.overload)
        if function_group is None:
            function_group = self.onnx_registry.get_op_functions(namespace=internal_opname.namespace, op_name=internal_opname.op_name, overload=None)
            if function_group is not None:
                op_full_name = internal_opname.qualified_name()
                diagnostic = diagnostic_context.inflight_diagnostic()
                diagnostic.warning('### The operator overload is not found in onnx registry!\nCannot find the operator overload in onnx registry, but the default overload is found. Please check the ONNX output carefully. \n')
                diagnostic.level = diagnostics.levels.WARNING
        if function_group is not None:
            function_group = self._filter_or_keep_complex(node, function_group, diagnostic_context)
            return function_group
        op_full_name = internal_opname.qualified_name()
        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Cannot find symbolic function for {op_full_name}, which should be registered under {node.target}.', unsupported_fx_node=node)
        diagnostic_context.log(diagnostic)
        raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)