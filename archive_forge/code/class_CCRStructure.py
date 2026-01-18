from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class CCRStructure(Structure):
    _fields_ = [('results', POINTER(CodeCompletionResult)), ('numResults', c_int)]

    def __len__(self):
        return self.numResults

    def __getitem__(self, key):
        if len(self) <= key:
            raise IndexError
        return self.results[key]