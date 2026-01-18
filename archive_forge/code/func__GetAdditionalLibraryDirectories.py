import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def _GetAdditionalLibraryDirectories(self, root, config, gyp_to_build_path):
    """Get and normalize the list of paths in AdditionalLibraryDirectories
        setting."""
    config = self._TargetConfig(config)
    libpaths = self._Setting((root, 'AdditionalLibraryDirectories'), config, default=[])
    libpaths = [os.path.normpath(gyp_to_build_path(self.ConvertVSMacros(p, config=config))) for p in libpaths]
    return ['/LIBPATH:"' + p + '"' for p in libpaths]