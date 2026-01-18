from abc import ABCMeta, abstractmethod
import sys
class ByteString(Sequence):
    """This unifies bytes and bytearray.

    XXX Should add all their methods.
    """
    __slots__ = ()