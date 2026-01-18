import sys
from past.utils import with_metaclass
class BaseOldDict(type):

    def __instancecheck__(cls, instance):
        return isinstance(instance, _builtin_dict)