from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class StorageClass(object):
    """
    Describes the storage class of a declaration
    """
    _kinds = []
    _name_map = None

    def __init__(self, value):
        if value >= len(StorageClass._kinds):
            StorageClass._kinds += [None] * (value - len(StorageClass._kinds) + 1)
        if StorageClass._kinds[value] is not None:
            raise ValueError('StorageClass already loaded')
        self.value = value
        StorageClass._kinds[value] = self
        StorageClass._name_map = None

    def from_param(self):
        return self.value

    @property
    def name(self):
        """Get the enumeration name of this storage class."""
        if self._name_map is None:
            self._name_map = {}
            for key, value in StorageClass.__dict__.items():
                if isinstance(value, StorageClass):
                    self._name_map[value] = key
        return self._name_map[self]

    @staticmethod
    def from_id(id):
        if id >= len(StorageClass._kinds) or not StorageClass._kinds[id]:
            raise ValueError('Unknown storage class %d' % id)
        return StorageClass._kinds[id]

    def __repr__(self):
        return 'StorageClass.%s' % (self.name,)