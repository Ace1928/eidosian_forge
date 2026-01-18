import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetAsmflags(self, config):
    """Returns the flags that need to be added to ml invocations."""
    config = self._TargetConfig(config)
    asmflags = []
    safeseh = self._Setting(('MASM', 'UseSafeExceptionHandlers'), config)
    if safeseh == 'true':
        asmflags.append('/safeseh')
    return asmflags