import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetNoImportLibrary(self, config):
    """If NoImportLibrary: true, ninja will not expect the output to include
        an import library."""
    config = self._TargetConfig(config)
    noimplib = self._Setting(('NoImportLibrary',), config)
    return noimplib == 'true'