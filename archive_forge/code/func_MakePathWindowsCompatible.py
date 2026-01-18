from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
def MakePathWindowsCompatible(path):
    """Converts invalid Windows characters to Unicode 'unsupported' character."""
    if re.search('^[A-Za-z]:', path):
        new_path = [path[:2]]
        start_index = 2
    else:
        new_path = []
        start_index = 0
    performed_conversion = False
    for i in range(start_index, len(path)):
        if path[i] in INVALID_WINDOWS_PATH_CHARACTERS:
            performed_conversion = True
            new_path.append('${}'.format(INVALID_WINDOWS_PATH_CHARACTERS.index(path[i])))
        else:
            new_path.append(path[i])
    new_path_string = ''.join(new_path)
    if performed_conversion:
        sys.stderr.write('WARNING: The following characters are invalid in Windows file and directory names: {}\nRenaming {} to {}'.format(''.join(INVALID_WINDOWS_PATH_CHARACTERS), path, new_path_string))
    return new_path_string