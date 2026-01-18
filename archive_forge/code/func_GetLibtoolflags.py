import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetLibtoolflags(self, configname):
    """Returns flags that need to be passed to the static linker.

    Args:
        configname: The name of the configuration to get ld flags for.
    """
    self.configname = configname
    libtoolflags = []
    for libtoolflag in self._Settings().get('OTHER_LDFLAGS', []):
        libtoolflags.append(libtoolflag)
    self.configname = None
    return libtoolflags