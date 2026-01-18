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
def __adapt__(self, obj):
    """Adapt an object to the receiver
        """
    if self.providedBy(obj):
        return obj
    for hook in adapter_hooks:
        adapter = hook(self, obj)
        if adapter is not None:
            return adapter
    return None