from numbers import Integral
from past.utils import PY2, with_metaclass
class BaseOldStr(type):

    def __instancecheck__(cls, instance):
        return isinstance(instance, _builtin_bytes)