import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def StringToMakefileVariable(string):
    """Convert a string to a value that is acceptable as a make variable name."""
    return re.sub('[^a-zA-Z0-9_]', '_', string)