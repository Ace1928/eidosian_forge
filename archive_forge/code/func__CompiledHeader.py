import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _CompiledHeader(self, lang, arch):
    assert self.compile_headers
    h = self.compiled_headers[lang]
    if arch:
        h += '.' + arch
    return h