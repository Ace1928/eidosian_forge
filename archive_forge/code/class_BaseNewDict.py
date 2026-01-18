import sys
from future.utils import with_metaclass
from future.types.newobject import newobject
class BaseNewDict(type):

    def __instancecheck__(cls, instance):
        if cls == newdict:
            return isinstance(instance, _builtin_dict)
        else:
            return issubclass(instance.__class__, cls)