import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetOutputName(self, config, expand_special):
    """Gets the explicitly overridden output name for a target or returns None
        if it's not overridden."""
    config = self._TargetConfig(config)
    type = self.spec['type']
    root = 'VCLibrarianTool' if type == 'static_library' else 'VCLinkerTool'
    output_file = self._Setting((root, 'OutputFile'), config)
    if output_file:
        output_file = expand_special(self.ConvertVSMacros(output_file, config=config))
    return output_file