import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetComputedDefines(self, config):
    """Returns the set of defines that are injected to the defines list based
        on other VS settings."""
    config = self._TargetConfig(config)
    defines = []
    if self._ConfigAttrib(['CharacterSet'], config) == '1':
        defines.extend(('_UNICODE', 'UNICODE'))
    if self._ConfigAttrib(['CharacterSet'], config) == '2':
        defines.append('_MBCS')
    defines.extend(self._Setting(('VCCLCompilerTool', 'PreprocessorDefinitions'), config, default=[]))
    return defines