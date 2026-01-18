from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class ArgumentsIterator(collections_abc.Sequence):

    def __init__(self, parent):
        self.parent = parent
        self.length = None

    def __len__(self):
        if self.length is None:
            self.length = conf.lib.clang_getNumArgTypes(self.parent)
        return self.length

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError('Must supply a non-negative int.')
        if key < 0:
            raise IndexError('Only non-negative indexes are accepted.')
        if key >= len(self):
            raise IndexError('Index greater than container length: %d > %d' % (key, len(self)))
        result = conf.lib.clang_getArgType(self.parent, key)
        if result.kind == TypeKind.INVALID:
            raise IndexError('Argument could not be retrieved.')
        return result