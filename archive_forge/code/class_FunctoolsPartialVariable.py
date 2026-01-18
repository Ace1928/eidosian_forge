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
class FunctoolsPartialVariable(VariableTracker):

    def __init__(self, func, args, keywords, original=None, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        assert isinstance(args, list)
        self.args = args
        assert isinstance(keywords, dict)
        self.keywords = keywords
        self.original = original

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        merged_args = self.args + args
        merged_kwargs = {**self.keywords, **kwargs}
        return self.func.call_function(tx, merged_args, merged_kwargs)

    def as_python_constant(self):
        if self.original:
            return self.original
        else:

            def get_val(v):
                if isinstance(v, variables.UserDefinedObjectVariable):
                    return v.value
                else:
                    return v.as_python_constant()
            return functools.partial(self.func.fn, *[get_val(arg) for arg in self.args], **{k: get_val(v) for k, v in self.keywords.items()})