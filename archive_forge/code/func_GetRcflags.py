import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetRcflags(self, config, gyp_to_ninja_path):
    """Returns the flags that need to be added to invocations of the resource
        compiler."""
    config = self._TargetConfig(config)
    rcflags = []
    rc = self._GetWrapper(self, self.msvs_settings[config], 'VCResourceCompilerTool', append=rcflags)
    rc('AdditionalIncludeDirectories', map=gyp_to_ninja_path, prefix='/I')
    rcflags.append('/I' + gyp_to_ninja_path('.'))
    rc('PreprocessorDefinitions', prefix='/d')
    rc('Culture', prefix='/l', map=lambda x: hex(int(x))[2:])
    return rcflags