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
class Attribute(Element):
    """Attribute descriptions
    """
    interface = None

    def _get_str_info(self):
        """Return extra data to put at the end of __str__."""
        return ''

    def __str__(self):
        of = ''
        if self.interface is not None:
            of = self.interface.__module__ + '.' + self.interface.__name__ + '.'
        return of + (self.__name__ or '<unknown>') + self._get_str_info()

    def __repr__(self):
        return '<{}.{} object at 0x{:x} {}>'.format(type(self).__module__, type(self).__name__, id(self), self)