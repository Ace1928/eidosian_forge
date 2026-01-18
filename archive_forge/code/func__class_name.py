from inspect import getfullargspec, getmro
import logging
from types import FunctionType
from .has_traits import HasTraits
def _class_name(self, cls):
    return cls.__name__