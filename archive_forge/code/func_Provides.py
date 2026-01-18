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
def Provides(*interfaces):
    """Declaration for an instance of *cls*.

       The correct signature is ``cls, *interfaces``.
       The *cls* is necessary to avoid the
       construction of inconsistent resolution orders.

      Instance declarations are shared among instances that have the same
      declaration. The declarations are cached in a weak value dictionary.
    """
    spec = InstanceDeclarations.get(interfaces)
    if spec is None:
        spec = ProvidesClass(*interfaces)
        InstanceDeclarations[interfaces] = spec
    return spec