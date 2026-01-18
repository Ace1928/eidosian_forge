import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetLibFlags(self, config, gyp_to_build_path):
    """Returns the flags that need to be added to lib commands."""
    config = self._TargetConfig(config)
    libflags = []
    lib = self._GetWrapper(self, self.msvs_settings[config], 'VCLibrarianTool', append=libflags)
    libflags.extend(self._GetAdditionalLibraryDirectories('VCLibrarianTool', config, gyp_to_build_path))
    lib('LinkTimeCodeGeneration', map={'true': '/LTCG'})
    lib('TargetMachine', map={'1': 'X86', '17': 'X64', '3': 'ARM'}, prefix='/MACHINE:')
    lib('AdditionalOptions')
    return libflags