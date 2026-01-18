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
class ClassProvides(Declaration, ClassProvidesBase):
    """Special descriptor for class ``__provides__``

    The descriptor caches the implementedBy info, so that
    we can get declarations for objects without instance-specific
    interfaces a bit quicker.
    """
    __slots__ = ('__args',)

    def __init__(self, cls, metacls, *interfaces):
        self._cls = cls
        self._implements = implementedBy(cls)
        self.__args = (cls, metacls) + interfaces
        Declaration.__init__(self, *self._add_interfaces_to_cls(interfaces, metacls))

    def __repr__(self):
        interfaces = (self._cls,) + self.__args[2:]
        ordered_names = self._argument_names_for_repr(interfaces)
        return 'directlyProvides({})'.format(ordered_names)

    def __reduce__(self):
        return (self.__class__, self.__args)
    __get__ = ClassProvidesBase.__get__