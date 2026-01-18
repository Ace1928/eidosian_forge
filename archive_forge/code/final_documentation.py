from typing import TypeVar
from types import FunctionType
Decorator to indicate that the decorated method is finalized and cannot be overridden.
    The decorator code is executed while loading class. Using this method
    should have minimal runtime performance implications.
    Currently, only methods with @overrides are checked.

    How to use:
    from overrides import final

    class SuperClass(object):
        @final
        def method(self):
          return 2

    class SubClass(SuperClass):
        @overrides
        def method(self): #causes an error
            return 1

    :raises  AssertionError if there exists a match in sub classes for the method name
    :return  method
    