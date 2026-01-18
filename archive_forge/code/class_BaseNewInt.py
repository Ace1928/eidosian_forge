from __future__ import division
import struct
from future.types.newbytes import newbytes
from future.types.newobject import newobject
from future.utils import PY3, isint, istext, isbytes, with_metaclass, native
class BaseNewInt(type):

    def __instancecheck__(cls, instance):
        if cls == newint:
            return isinstance(instance, (int, long))
        else:
            return issubclass(instance.__class__, cls)