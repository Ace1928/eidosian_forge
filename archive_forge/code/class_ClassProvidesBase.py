import sys
import weakref
from types import FunctionType
from types import MethodType
from types import ModuleType
from zope.interface._compat import _use_c_impl
from zope.interface.interface import Interface
from zope.interface.interface import InterfaceClass
from zope.interface.interface import NameAndModuleComparisonMixin
from zope.interface.interface import Specification
from zope.interface.interface import SpecificationBase
@_use_c_impl
class ClassProvidesBase(SpecificationBase):
    __slots__ = ('_cls', '_implements')

    def __get__(self, inst, cls):
        if cls is self._cls:
            if inst is None:
                return self
            return self._implements
        raise AttributeError('__provides__')