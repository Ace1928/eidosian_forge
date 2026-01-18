import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetCflagsCC(self, configname):
    """Returns flags that need to be added to .cc, and .mm compilations."""
    self.configname = configname
    cflags_cc = []
    clang_cxx_language_standard = self._Settings().get('CLANG_CXX_LANGUAGE_STANDARD')
    if clang_cxx_language_standard:
        cflags_cc.append('-std=%s' % clang_cxx_language_standard)
    self._Appendf(cflags_cc, 'CLANG_CXX_LIBRARY', '-stdlib=%s')
    if self._Test('GCC_ENABLE_CPP_RTTI', 'NO', default='YES'):
        cflags_cc.append('-fno-rtti')
    if self._Test('GCC_ENABLE_CPP_EXCEPTIONS', 'NO', default='YES'):
        cflags_cc.append('-fno-exceptions')
    if self._Test('GCC_INLINES_ARE_PRIVATE_EXTERN', 'YES', default='NO'):
        cflags_cc.append('-fvisibility-inlines-hidden')
    if self._Test('GCC_THREADSAFE_STATICS', 'NO', default='YES'):
        cflags_cc.append('-fno-threadsafe-statics')
    if self._Test('GCC_WARN_ABOUT_INVALID_OFFSETOF_MACRO', 'NO', default='YES'):
        cflags_cc.append('-Wno-invalid-offsetof')
    other_ccflags = []
    for flag in self._Settings().get('OTHER_CPLUSPLUSFLAGS', ['$(inherited)']):
        if flag in ('$inherited', '$(inherited)', '${inherited}'):
            flag = '$OTHER_CFLAGS'
        if flag in ('$OTHER_CFLAGS', '$(OTHER_CFLAGS)', '${OTHER_CFLAGS}'):
            other_ccflags += self._Settings().get('OTHER_CFLAGS', [])
        else:
            other_ccflags.append(flag)
    cflags_cc += other_ccflags
    self.configname = None
    return cflags_cc