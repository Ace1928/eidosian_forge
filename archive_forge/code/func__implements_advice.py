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
def _implements_advice(cls):
    interfaces, do_classImplements = cls.__dict__['__implements_advice_data__']
    del cls.__implements_advice_data__
    do_classImplements(cls, *interfaces)
    return cls