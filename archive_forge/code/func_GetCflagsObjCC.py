import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetCflagsObjCC(self, configname):
    """Returns flags that need to be added to .mm compilations."""
    self.configname = configname
    cflags_objcc = []
    self._AddObjectiveCGarbageCollectionFlags(cflags_objcc)
    self._AddObjectiveCARCFlags(cflags_objcc)
    self._AddObjectiveCMissingPropertySynthesisFlags(cflags_objcc)
    if self._Test('GCC_OBJC_CALL_CXX_CDTORS', 'YES', default='NO'):
        cflags_objcc.append('-fobjc-call-cxx-cdtors')
    self.configname = None
    return cflags_objcc