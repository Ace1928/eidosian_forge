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
def _calculate_sro(self):
    """
        Calculate and return the resolution order for this object, using its ``__bases__``.

        Ensures that ``Interface`` is always the last (lowest priority) element.
        """
    sro = self._do_calculate_ro(base_mros={b: b.__sro__ for b in self.__bases__})
    root = self._ROOT
    if root is not None and sro and (sro[-1] is not root):
        sro = [x for x in sro if x is not root]
        sro.append(root)
    return sro