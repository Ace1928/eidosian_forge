import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def Sourceify(path):
    """Convert a path to its source directory form."""
    if '$(' in path:
        return path
    if os.path.isabs(path):
        return path
    return srcdir_prefix + path