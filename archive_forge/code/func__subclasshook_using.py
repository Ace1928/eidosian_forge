from abc import ABCMeta, abstractmethod
from collections.abc import Callable
@classmethod
def _subclasshook_using(cls, parent, C):
    return cls is parent and all((_hasattr(C, attr) for attr in cls.__required_attributes__)) or NotImplemented