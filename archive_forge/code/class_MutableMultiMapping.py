import abc
import sys
import types
from collections.abc import Mapping, MutableMapping
class MutableMultiMapping(MultiMapping, MutableMapping):

    @abc.abstractmethod
    def add(self, key, value):
        raise NotImplementedError

    @abc.abstractmethod
    def extend(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def popone(self, key, default=None):
        raise KeyError

    @abc.abstractmethod
    def popall(self, key, default=None):
        raise KeyError