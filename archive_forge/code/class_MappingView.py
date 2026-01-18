from abc import ABCMeta, abstractmethod
import sys
class MappingView(Sized):
    __slots__ = ('_mapping',)

    def __init__(self, mapping):
        self._mapping = mapping

    def __len__(self):
        return len(self._mapping)

    def __repr__(self):
        return '{0.__class__.__name__}({0._mapping!r})'.format(self)
    __class_getitem__ = classmethod(GenericAlias)