import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetPDBName(self, config, expand_special, default):
    """Gets the explicitly overridden pdb name for a target or returns
        default if it's not overridden, or if no pdb will be generated."""
    config = self._TargetConfig(config)
    output_file = self._Setting(('VCLinkerTool', 'ProgramDatabaseFile'), config)
    generate_debug_info = self._Setting(('VCLinkerTool', 'GenerateDebugInformation'), config)
    if generate_debug_info == 'true':
        if output_file:
            return expand_special(self.ConvertVSMacros(output_file, config=config))
        else:
            return default
    else:
        return None