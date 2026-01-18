import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetPchBuildCommands(self, arch=None):
    """Returns [(path_to_gch, language_flag, language, header)].
    |path_to_gch| and |header| are relative to the build directory.
    """
    if not self.header or not self.compile_headers:
        return []
    return [(self._Gch('c', arch), '-x c-header', 'c', self.header), (self._Gch('cc', arch), '-x c++-header', 'cc', self.header), (self._Gch('m', arch), '-x objective-c-header', 'm', self.header), (self._Gch('mm', arch), '-x objective-c++-header', 'mm', self.header)]