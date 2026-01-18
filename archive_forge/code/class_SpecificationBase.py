import sys
import weakref
from types import FunctionType
from types import MethodType
from typing import Union
from zope.interface import ro
from zope.interface._compat import _use_c_impl
from zope.interface.exceptions import Invalid
from zope.interface.ro import ro as calculate_ro
from zope.interface.declarations import implementedBy
from zope.interface.declarations import providedBy
from zope.interface.exceptions import BrokenImplementation
from zope.interface.exceptions import InvalidInterface
from zope.interface.declarations import _empty
@_use_c_impl
class SpecificationBase:
    __slots__ = ('_implied', '_dependents', '_bases', '_v_attrs', '__iro__', '__sro__', '__weakref__')

    def providedBy(self, ob):
        """Is the interface implemented by an object
        """
        spec = providedBy(ob)
        return self in spec._implied

    def implementedBy(self, cls):
        """Test whether the specification is implemented by a class or factory.

        Raise TypeError if argument is neither a class nor a callable.
        """
        spec = implementedBy(cls)
        return self in spec._implied

    def isOrExtends(self, interface):
        """Is the interface the same as or extend the given interface
        """
        return interface in self._implied
    __call__ = isOrExtends