from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class FixItIterator(object):

    def __init__(self, diag):
        self.diag = diag

    def __len__(self):
        return int(conf.lib.clang_getDiagnosticNumFixIts(self.diag))

    def __getitem__(self, key):
        range = SourceRange()
        value = conf.lib.clang_getDiagnosticFixIt(self.diag, key, byref(range))
        if len(value) == 0:
            raise IndexError
        return FixIt(range, value)