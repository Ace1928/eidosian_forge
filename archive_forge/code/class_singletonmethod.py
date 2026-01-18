import copy
import types
from itertools import count
class singletonmethod:
    """
    For Declarative subclasses, this decorator will call the method
    on the cls.singleton() object if called as a class method (or
    as normal if called as an instance method).
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls=None):
        if obj is None:
            obj = cls.singleton()
        return types.MethodType(self.func, obj)