from numbers import Number
from future.utils import PY3, istext, with_metaclass, isnewbytes
from future.types import no, issubset
from future.types.newobject import newobject
class BaseNewStr(type):

    def __instancecheck__(cls, instance):
        if cls == newstr:
            return isinstance(instance, unicode)
        else:
            return issubclass(instance.__class__, cls)