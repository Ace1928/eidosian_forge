from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class DiagnosticsItr(object):

    def __init__(self, ccr):
        self.ccr = ccr

    def __len__(self):
        return int(conf.lib.clang_codeCompleteGetNumDiagnostics(self.ccr))

    def __getitem__(self, key):
        return conf.lib.clang_codeCompleteGetDiagnostic(self.ccr, key)