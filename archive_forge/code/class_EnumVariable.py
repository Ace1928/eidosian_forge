import operator
from typing import Dict, List
import torch
from torch._dynamo.source import GetItemSource
from .. import variables
from ..exc import unimplemented, UserError, UserErrorType
from ..guards import GuardBuilder, install_guard
from ..utils import np
from .base import typestr, VariableTracker
class EnumVariable(VariableTracker):

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def as_proxy(self):
        return self.value

    def __str__(self):
        return f'EnumVariable({type(self.value)})'

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value

    def const_getattr(self, tx, name):
        member = getattr(self.value, name)
        if callable(member):
            raise NotImplementedError()
        return member