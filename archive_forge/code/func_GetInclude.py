import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetInclude(self, lang, arch=None):
    """Gets the cflags to include the prefix header for language |lang|."""
    if self.compile_headers and lang in self.compiled_headers:
        return '-include %s' % self._CompiledHeader(lang, arch)
    elif self.header:
        return '-include %s' % self.header
    else:
        return ''