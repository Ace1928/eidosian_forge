from .api import NoDefault, Invalid
from .compound import CompoundValidator, from_python
class _IfMissing:

    def __get__(self, obj, cls=None):
        if obj is None:
            return []
        if obj._if_missing is ForEach._if_missing:
            return []
        return obj._if_missing

    def __set__(self, obj, value):
        obj._if_missing = value

    def __delete__(self, obj):
        obj._if_missing = NoDefault