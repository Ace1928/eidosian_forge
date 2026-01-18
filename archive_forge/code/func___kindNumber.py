from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def __kindNumber(self):
    if self.__kindNumberCache == -1:
        self.__kindNumberCache = conf.lib.clang_getCompletionChunkKind(self.cs, self.key)
    return self.__kindNumberCache