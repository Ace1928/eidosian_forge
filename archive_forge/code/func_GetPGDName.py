import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetPGDName(self, config, expand_special):
    """Gets the explicitly overridden pgd name for a target or returns None
        if it's not overridden."""
    config = self._TargetConfig(config)
    output_file = self._Setting(('VCLinkerTool', 'ProfileGuidedDatabase'), config)
    if output_file:
        output_file = expand_special(self.ConvertVSMacros(output_file, config=config))
    return output_file