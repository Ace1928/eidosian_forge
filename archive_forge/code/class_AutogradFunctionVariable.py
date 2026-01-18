import collections
import dataclasses
import functools
import inspect
import itertools
import operator
import sys
import types
from typing import Dict, List
import torch._C
import torch._numpy as tnp
from .. import config, polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, ODictGetItemSource, TypeSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .dicts import DefaultDictVariable
from .functions import (
from .user_defined import UserDefinedObjectVariable
class AutogradFunctionVariable(VariableTracker):
    """represents a torch.autograd.Function subclass"""

    def __init__(self, fn_cls, **kwargs):
        super().__init__(**kwargs)
        self.fn_cls = fn_cls

    def call_apply(self, tx, args, kwargs):
        requires_grad = False

        def visit(node):
            nonlocal requires_grad
            if isinstance(node, variables.TensorVariable):
                if node.requires_grad is not False:
                    requires_grad = True
            if isinstance(node, variables.NNModuleVariable):
                if node.is_training(tx):
                    requires_grad = True
            return node
        VariableTracker.apply(visit, (args, kwargs))
        ctx = AutogradFunctionContextVariable.create(tx)
        args = [ctx, *args]
        if requires_grad and torch.is_grad_enabled() and config.capture_autograd_function:
            if self.fn_cls.setup_context != torch.autograd.function._SingleLevelFunction.setup_context:
                unimplemented('NYI - autograd.Function with custom setup_context method')
            vjp_fn = self.fn_cls.vjp
            if vjp_fn is not torch.autograd.Function.vjp:
                unimplemented('NYI - User defind vjp')
            jvp_fn = self.fn_cls.jvp
            if jvp_fn is not torch.autograd.Function.jvp:
                unimplemented('NYI - User defind jvp')
            from .higher_order_ops import safe_or_raise_always_restore, TorchHigherOrderOperatorVariable
            trampoline_autograd_apply = produce_trampoline_autograd_apply(self.fn_cls)
            trampoline_autograd_fwd = produce_trampoline_autograd_fwd(self.fn_cls)
            trampoline_autograd_bwd = produce_trampoline_autograd_bwd(self.fn_cls)
            graph_checkpoint, checkpoint = (tx.output.graph, tx.copy_graphstate())
            module_source = AttrSource(tx.import_source(self.fn_cls.__module__), self.fn_cls.__name__)
            fwd_bwd_tracer = torch._dynamo.output_graph.SubgraphTracer(tx.output, parent=tx.output.current_tracer, source_target='autograd.Function')
            higher_order_autograd_fn = TorchHigherOrderOperatorVariable.make(trampoline_autograd_fwd, source=AttrSource(module_source, 'forward'), fwd_bwd_tracer=fwd_bwd_tracer)
            speculated_fwd_result = higher_order_autograd_fn.call_function(tx, args, kwargs)
            if isinstance(speculated_fwd_result, variables.TupleVariable):
                bwd_args = [ctx, *speculated_fwd_result.items]
            else:
                bwd_args = [ctx, speculated_fwd_result]
            safe_or_raise_always_restore(tx, graph_checkpoint, checkpoint, TorchHigherOrderOperatorVariable.make(trampoline_autograd_bwd, source=AttrSource(module_source, 'backward'), fwd_bwd_tracer=fwd_bwd_tracer), bwd_args)
            args = args[1:]
            return TorchHigherOrderOperatorVariable.make(trampoline_autograd_apply, fwd_bwd_tracer=None).call_function(tx, args, kwargs)
        if self.source:
            source = AttrSource(AttrSource(self.source, '__class__'), 'forward')
        else:
            source = None
        fn = self.fn_cls.forward
        if isinstance(fn, types.FunctionType):
            return variables.UserFunctionVariable(fn, source=source).call_function(tx, args, kwargs)
        elif isinstance(fn, types.MethodType):
            return variables.UserMethodVariable(fn.__func__, variables.UserDefinedClassVariable(self.fn_cls), source=source).call_function(tx, args, kwargs)
        else:
            unimplemented(f'non-function or method in subclass of torch.autograd.Function: {fn}')

    def call_function(self, tx, args, kwargs):
        return AutogradFunctionVariable(self.fn_cls)

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]'):
        from ..allowed_functions import is_user_defined_allowed
        from .builder import wrap_fx_proxy
        if name == 'apply':
            if is_user_defined_allowed(self.fn_cls):
                trampoline_autograd_apply = produce_trampoline_autograd_apply(self.fn_cls)
                return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', trampoline_autograd_apply, *proxy_args_kwargs(args, kwargs)))
            else:
                return self.call_apply(tx, args, kwargs)
        elif name == 'backward':
            with tx.strict_translation_mode():
                if isinstance(self.fn_cls.backward, types.FunctionType):
                    backward = UserFunctionVariable(self.fn_cls.backward)
                elif isinstance(self.fn_cls.backward, types.MethodType):
                    backward = UserMethodVariable(self.fn_cls.backward.__func__, variables.UserDefinedClassVariable(self.fn_cls))
                    args = [backward.obj] + args
                else:
                    unimplemented(f'backward is a non-function or method: {self.fn_cls.backward}')
                return tx.inline_call(tx, backward, args, kwargs)
        elif name == 'forward':
            if isinstance(self.fn_cls.forward, types.FunctionType):
                forward = UserFunctionVariable(self.fn_cls.forward)
            elif isinstance(self.fn_cls.forward, types.MethodType):
                forward = UserMethodVariable(self.fn_cls.forward.__func__, variables.UserDefinedClassVariable(self.fn_cls))
                args = [forward.obj] + args
            else:
                unimplemented(f'forward is a non-function or method: {self.fn_cls.forward}')
            return tx.inline_call(tx, forward, args, kwargs)
        else:
            unimplemented(f'Unsupported method: {name}')