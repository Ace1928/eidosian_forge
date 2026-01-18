from numbers import Integral
import string
import copy
from future.utils import istext, isbytes, PY2, PY3, with_metaclass
from future.types import no, issubset
from future.types.newobject import newobject
class BaseNewBytes(type):

    def __instancecheck__(cls, instance):
        if cls == newbytes:
            return isinstance(instance, _builtin_bytes)
        else:
            return issubclass(instance.__class__, cls)