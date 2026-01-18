import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def IsLinkIncremental(self, config):
    """Returns whether the target should be linked incrementally."""
    config = self._TargetConfig(config)
    link_inc = self._Setting(('VCLinkerTool', 'LinkIncremental'), config)
    return link_inc != '1'