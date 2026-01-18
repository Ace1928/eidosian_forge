import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetMapFileName(self, config, expand_special):
    """Gets the explicitly overridden map file name for a target or returns None
        if it's not set."""
    config = self._TargetConfig(config)
    map_file = self._Setting(('VCLinkerTool', 'MapFileName'), config)
    if map_file:
        map_file = expand_special(self.ConvertVSMacros(map_file, config=config))
    return map_file