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
class Provides(Declaration):
    """Implement ``__provides__``, the instance-specific specification

    When an object is pickled, we pickle the interfaces that it implements.
    """

    def __init__(self, cls, *interfaces):
        self.__args = (cls,) + interfaces
        self._cls = cls
        Declaration.__init__(self, *self._add_interfaces_to_cls(interfaces, cls))
    _v_module_names = ()

    def __repr__(self):
        function_name = 'directlyProvides'
        if self._cls is ModuleType and self._v_module_names:
            providing_on_module = True
            interfaces = self.__args[1:]
        else:
            providing_on_module = False
            interfaces = (self._cls,) + self.__bases__
        ordered_names = self._argument_names_for_repr(interfaces)
        if providing_on_module:
            mod_names = self._v_module_names
            if len(mod_names) == 1:
                mod_names = 'sys.modules[%r]' % mod_names[0]
            ordered_names = '{}, '.format(mod_names) + ordered_names
        return '{}({})'.format(function_name, ordered_names)

    def __reduce__(self):
        return (Provides, self.__args)
    __module__ = 'zope.interface'

    def __get__(self, inst, cls):
        """Make sure that a class __provides__ doesn't leak to an instance
        """
        if inst is None and cls is self._cls:
            return self
        raise AttributeError('__provides__')