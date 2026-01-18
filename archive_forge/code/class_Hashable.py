from abc import ABCMeta, abstractmethod
import sys
class Hashable(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def __hash__(self):
        return 0

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Hashable:
            return _check_methods(C, '__hash__')
        return NotImplemented