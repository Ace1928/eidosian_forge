import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _Setting(self, path, config, default=None, prefix='', append=None, map=None):
    """_GetAndMunge for msvs_settings."""
    return self._GetAndMunge(self.msvs_settings[config], path, default, prefix, append, map)