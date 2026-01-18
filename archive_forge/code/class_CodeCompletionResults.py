from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class CodeCompletionResults(ClangObject):

    def __init__(self, ptr):
        assert isinstance(ptr, POINTER(CCRStructure)) and ptr
        self.ptr = self._as_parameter_ = ptr

    def from_param(self):
        return self._as_parameter_

    def __del__(self):
        conf.lib.clang_disposeCodeCompleteResults(self)

    @property
    def results(self):
        return self.ptr.contents

    @property
    def diagnostics(self):

        class DiagnosticsItr(object):

            def __init__(self, ccr):
                self.ccr = ccr

            def __len__(self):
                return int(conf.lib.clang_codeCompleteGetNumDiagnostics(self.ccr))

            def __getitem__(self, key):
                return conf.lib.clang_codeCompleteGetDiagnostic(self.ccr, key)
        return DiagnosticsItr(self)