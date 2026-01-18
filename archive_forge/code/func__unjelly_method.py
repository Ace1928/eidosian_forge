import copy
import datetime
import decimal
import types
import warnings
from functools import reduce
from zope.interface import implementer
from incremental import Version
from twisted.persisted.crefutil import (
from twisted.python.compat import nativeString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.reflect import namedAny, namedObject, qual
from twisted.spread.interfaces import IJellyable, IUnjellyable
def _unjelly_method(self, rest):
    """
        (internal) Unjelly a method.
        """
    im_name = rest[0]
    im_self = self.unjelly(rest[1])
    im_class = self.unjelly(rest[2])
    if not isinstance(im_class, type):
        raise InsecureJelly('Method found with non-class class.')
    if im_name in im_class.__dict__:
        if im_self is None:
            im = getattr(im_class, im_name)
        elif isinstance(im_self, NotKnown):
            im = _InstanceMethod(im_name, im_self, im_class)
        else:
            im = types.MethodType(im_class.__dict__[im_name], im_self, *[im_class] * False)
    else:
        raise TypeError('instance method changed')
    return im