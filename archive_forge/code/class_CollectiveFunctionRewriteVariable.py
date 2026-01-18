import functools
import inspect
import itertools
import types
from typing import Dict, List
import torch
from .. import variables
from ..bytecode_transformation import create_call_function, create_rot_n
from ..exc import unimplemented, Unsupported
from ..source import AttrSource, ConstantSource, DefaultsSource, GetItemSource
from ..utils import make_cell
from .base import typestr, VariableTracker
class CollectiveFunctionRewriteVariable(UserFunctionVariable):
    """
    Some of the torch.distributed.* collective APIs are possible to rewrite to 'traceable' collectives.

    This class provides both a way to check if a function is remappable, and perform the remapping.

    In the case that a function is 'remappable' but only for some combinations of call-time arguments,
    we check the args at `call_function` time and fall back to graph-breaking if needed.  This is no worse
    than status-quo as we currently graph-break on all distributed.* collectives.
    """

    def __init__(self, fn, *, replacement_var, **kwargs):
        super().__init__(fn, **kwargs)
        assert isinstance(replacement_var, UserFunctionVariable)
        self.replacement_var = replacement_var

    @staticmethod
    def create(tx, old_fn, source, **options):
        new_fn, new_source = CollectiveFunctionRewriteVariable.rewrite(tx, old_fn)
        return CollectiveFunctionRewriteVariable(old_fn, replacement_var=UserFunctionVariable(new_fn, source=new_source, **options), source=source, **options)

    @staticmethod
    def can_rewrite(variable):
        return inspect.isfunction(variable) and variable in _traceable_collective_remaps()

    @staticmethod
    def rewrite(tx, fn):
        new_fn = _traceable_collective_remaps()[fn]
        return (new_fn, _traceable_collectives_source(tx, new_fn))

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if kwargs.get('async_op', False):
            unimplemented(f"CollectiveFunctionRewriteVariable can't support async_op=True for {self.fn}")
        return self.replacement_var.call_function(tx, args, kwargs)