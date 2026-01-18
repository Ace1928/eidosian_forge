import operator
import sys
from functools import reduce
from importlib import import_module
from types import ModuleType
def _default_cls_attr(name, type_, cls_value):

    def __new__(cls, getter):
        instance = type_.__new__(cls, cls_value)
        instance.__getter = getter
        return instance

    def __get__(self, obj, cls=None):
        return self.__getter(obj) if obj is not None else self
    return type(name, (type_,), {'__new__': __new__, '__get__': __get__})