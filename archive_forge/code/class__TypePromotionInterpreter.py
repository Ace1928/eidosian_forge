from __future__ import annotations
import abc
import dataclasses
import inspect
import logging
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence, Set
import torch
import torch._ops
import torch.fx
import torch.fx.traceback as fx_traceback
from torch import _prims_common, _refs
from torch._prims_common import (
from torch._refs import linalg as _linalg_refs, nn as _nn_refs, special as _special_refs
from torch._refs.nn import functional as _functional_refs
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.fx.node import Node  # noqa: F401
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics, type_utils as fx_type_utils
from torch.utils import _python_dispatch, _pytree
class _TypePromotionInterpreter(torch.fx.Interpreter):
    """Interpreter that inserts type promotion for each node."""

    def __init__(self, diagnostic_context: diagnostics.DiagnosticContext, module: torch.fx.GraphModule, type_promotion_table: TypePromotionTable):
        super().__init__(module)
        self.diagnostic_context = diagnostic_context
        self.type_promotion_table = type_promotion_table

    def _run_node_and_set_meta(self, node) -> Any:
        """Run node and set meta according to `fx_traceback.get_current_meta()`.

        This should be used on new nodes or nodes that have been modified.
        By default `Interpreter.run_node` does not update `node.meta`.
        Set `node.meta` to the current meta, except for `node.meta["val"]`, which is
        recomputed.
        """
        out = super().run_node(node)
        self.env[node] = out
        node.meta.update(((k, v) for k, v in fx_traceback.get_current_meta().items() if k not in node.meta))
        node.meta['val'] = proxy_tensor.extract_val(out)
        return out

    @_beartype.beartype
    def _create_node(self, graph: torch.fx.Graph, op_type: str, target: torch.fx.node.Target, args: tuple, kwargs: dict) -> torch.fx.Node:
        """Create a node and set its metadata."""
        assert op_type in ('call_function', 'call_method', 'get_attr', 'call_module', 'placeholder', 'output'), f'Unexpected op_type: {op_type}'
        node = getattr(graph, op_type)(target, args, kwargs)
        self._run_node_and_set_meta(node)
        return node

    @_beartype.beartype
    def _rerun_node_after_type_promotion(self, diagnostic: diagnostics.Diagnostic, node: torch.fx.Node, expected_out_dtype: torch.dtype) -> None:
        """Rerun a node after type promotion and update node.meta["val"] with the output value."""
        node_val = node.meta.get('val', None)
        assert node_val is not None, f"Node {node} node.meta['val'] is not set."
        args, kwargs = self.fetch_args_kwargs_from_env(node)
        target = node.target
        assert isinstance(target, torch._ops.OpOverload), f'Expected OpOverload, got {type(target)}'
        node.target = find_compatible_op_overload(target.overloadpacket, args, kwargs)
        new_node_val = self._run_node_and_set_meta(node)
        assert isinstance(new_node_val, type(node_val)), f'run_node output type should not change between runs. Got {type(new_node_val)}, expect {type(node_val)}.'
        if isinstance(node_val, torch.Tensor):
            prev_node_dtype = node_val.dtype
            assert prev_node_dtype == expected_out_dtype, f"node.meta['val'].dtype({prev_node_dtype}) does not agree with type promotion rule({expected_out_dtype})."
            if new_node_val.dtype != expected_out_dtype:
                graph = node.graph
                with graph.inserting_after(node):
                    output_cast_node = self._create_node(graph, 'call_function', torch.ops.prims.convert_element_type.default, (node,), {'dtype': expected_out_dtype})
                    node.replace_all_uses_with(output_cast_node)
                    output_cast_node.args = (node,)
                    diagnostic.info("Node '%s' output dtype becomes %s due to op math. Cast back to %s.", node, new_node_val.dtype, expected_out_dtype)
        elif fx_type_utils.is_torch_symbolic_type(node_val):
            raise NotImplementedError('Type promotion does not support node output of sym types.')
        elif isinstance(node_val, (list, tuple)):
            raise NotImplementedError('Type promotion does not support node output of list or tuple.')
        else:
            raise RuntimeError(f'Unexpected node output type: {type(node_val)}.')

    @_beartype.beartype
    def _maybe_promote_arg(self, diagnostic: diagnostics.Diagnostic, node: torch.fx.Node, fx_arg: torch.fx.node.Argument, dtype: Optional[torch.dtype]) -> torch.fx.node.Argument:
        """Promote fx_arg to dtype if necessary."""
        if dtype is None:
            diagnostic.info('Argument %s is not promoted. Not mentioned by type promotion rule.', fx_arg)
            return fx_arg
        if isinstance(fx_arg, torch.fx.Node):
            arg_val = self.env[fx_arg]
            if isinstance(arg_val, torch.Tensor):
                if (old_dtype := arg_val.dtype) != dtype:
                    graph = node.graph
                    with graph.inserting_before(node):
                        diagnostic.info('Argument %s(%s) is promoted to %s.', fx_arg, old_dtype, dtype)
                        return self._create_node(graph, 'call_function', torch.ops.prims.convert_element_type.default, (fx_arg,), {'dtype': dtype})
                diagnostic.info('Argument %s is not promoted. Already %s.', fx_arg, dtype)
                return fx_arg
            elif fx_type_utils.is_torch_symbolic_type(arg_val):
                arg_type = type(arg_val)
                equivalent_dtype = fx_type_utils.from_scalar_type_to_torch_dtype(arg_type)
                assert equivalent_dtype is not None, f'Unexpected arg_type: {arg_type}'
                if equivalent_dtype != dtype:
                    graph = node.graph
                    with graph.inserting_before(node):
                        diagnostic.info('Argument %s(Scalar of equivalent dtype: %s) is promoted to %s.', fx_arg, equivalent_dtype, dtype)
                        return self._create_node(graph, 'call_function', torch.ops.aten.scalar_tensor.default, (fx_arg,), {'dtype': dtype})
                diagnostic.info('Argument %s is not promoted. Already %s.', fx_arg, dtype)
                return fx_arg
        elif (equivalent_dtype := fx_type_utils.from_scalar_type_to_torch_dtype(type(fx_arg))) is not None:
            if equivalent_dtype != dtype:
                graph = node.graph
                with graph.inserting_before(node):
                    diagnostic.info('Argument %s(Scalar of equivalent dtype: %s) is promoted to %s.', fx_arg, equivalent_dtype, dtype)
                    return self._create_node(graph, 'call_function', torch.ops.aten.scalar_tensor.default, (fx_arg,), {'dtype': dtype})
            diagnostic.info('Argument %s is not promoted. Already %s.', fx_arg, dtype)
            return fx_arg
        elif isinstance(fx_arg, (tuple, list)):
            diagnostic.info('Argument %s is a tuple/list. Promoting each element.', fx_arg)
            return type(fx_arg)((self._maybe_promote_arg(diagnostic, node, fx_arg_elem, dtype) for fx_arg_elem in fx_arg))
        raise NotImplementedError(f'Unknown fx arg type: {type(fx_arg)}')

    @_beartype.beartype
    def _maybe_promote_node(self, diagnostic: diagnostics.Diagnostic, node: torch.fx.Node, rule: TypePromotionRule) -> torch.fx.Node:
        """Promote node inputs and outputs according to type promotion rule."""
        args, kwargs = self.fetch_args_kwargs_from_env(node)
        type_promotion_info = rule.preview_type_promotion(args, kwargs)
        new_args = []
        new_kwargs = {}
        for i, arg in enumerate(node.args):
            new_args.append(self._maybe_promote_arg(diagnostic, node, arg, type_promotion_info.args_dtypes.get(i, None)))
        for name, arg in node.kwargs.items():
            new_kwargs[name] = self._maybe_promote_arg(diagnostic, node, arg, type_promotion_info.kwargs_dtypes.get(name, None))
        new_args = tuple(new_args)
        if node.args != new_args or node.kwargs != new_kwargs:
            diagnostic.message = f'Applied type promotion for {node}. '
            node.args = new_args
            node.kwargs = new_kwargs
            self._rerun_node_after_type_promotion(diagnostic, node, type_promotion_info.out_dtype)
        else:
            diagnostic.message = f'Type promotion not needed for {node}. '
        return node

    @diagnostics.diagnose_call(rule=diagnostics.rules.fx_node_insert_type_promotion, level=diagnostics.levels.NONE)
    def run_node(self, node: torch.fx.Node) -> Any:
        """This method is an override which inserts type promotion nodes as needed.

        For each `call_function` node, an initial check is conducted to determine if a type
        promotion rule is applicable. If a relevant rule exists, type casting nodes are
        introduced for the corresponding arguments. The OpOverload of the node is updated
        to one that accommodates the promoted types. Should the output type be different,
        type casting node is inserted for this output.

        The call `super().run_node(node)` is guaranteed to be invoked for each node.
        In the case of new or modified nodes, the result of `super().run_node(node)` is
        used to update its `node.meta["val"]` value.
        """
        diagnostic = self.diagnostic_context.inflight_diagnostic()
        with self._set_current_node(node):
            if node.op != 'call_function':
                diagnostic.message = f'Skipped {node}: not a call_function.'
            elif (rule := get_type_promotion_rule(diagnostic, node, self.type_promotion_table)):
                self._maybe_promote_node(diagnostic, node, rule)
        return super().run_node(node)