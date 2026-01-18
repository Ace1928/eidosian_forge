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
class _ImmutableDeclaration(Declaration):
    __slots__ = ()
    __instance = None

    def __new__(cls):
        if _ImmutableDeclaration.__instance is None:
            _ImmutableDeclaration.__instance = object.__new__(cls)
        return _ImmutableDeclaration.__instance

    def __reduce__(self):
        return '_empty'

    @property
    def __bases__(self):
        return ()

    @__bases__.setter
    def __bases__(self, new_bases):
        if new_bases != ():
            raise TypeError('Cannot set non-empty bases on shared empty Declaration.')

    @property
    def dependents(self):
        return {}
    changed = subscribe = unsubscribe = lambda self, _ignored: None

    def interfaces(self):
        return iter(())

    def extends(self, interface, strict=True):
        return interface is self._ROOT

    def get(self, name, default=None):
        return default

    def weakref(self, callback=None):
        return _ImmutableDeclaration

    @property
    def _v_attrs(self):
        return {}

    @_v_attrs.setter
    def _v_attrs(self, new_attrs):
        pass