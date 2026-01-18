import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetCompilerPdbName(self, config, expand_special):
    """Get the pdb file name that should be used for compiler invocations, or
        None if there's no explicit name specified."""
    config = self._TargetConfig(config)
    pdbname = self._Setting(('VCCLCompilerTool', 'ProgramDataBaseFileName'), config)
    if pdbname:
        pdbname = expand_special(self.ConvertVSMacros(pdbname))
    return pdbname