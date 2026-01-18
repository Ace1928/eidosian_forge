import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetFlagsModifications(self, input, output, implicit, command, cflags_c, cflags_cc, expand_special):
    """Get the modified cflags and implicit dependencies that should be used
        for the pch compilation step."""
    if input == self.pch_source:
        pch_output = ['/Yc' + self._PchHeader()]
        if command == 'cxx':
            return ([('cflags_cc', map(expand_special, cflags_cc + pch_output))], self.output_obj, [])
        elif command == 'cc':
            return ([('cflags_c', map(expand_special, cflags_c + pch_output))], self.output_obj, [])
    return ([], output, implicit)