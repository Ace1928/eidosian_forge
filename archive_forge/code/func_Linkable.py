import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def Linkable(filename):
    """Return true if the file is linkable (should be on the link line)."""
    return filename.endswith('.o')