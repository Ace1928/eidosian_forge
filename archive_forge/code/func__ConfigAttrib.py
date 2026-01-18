import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _ConfigAttrib(self, path, config, default=None, prefix='', append=None, map=None):
    """_GetAndMunge for msvs_configuration_attributes."""
    return self._GetAndMunge(self.msvs_configuration_attributes[config], path, default, prefix, append, map)