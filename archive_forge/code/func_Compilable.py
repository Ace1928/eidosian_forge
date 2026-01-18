import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def Compilable(filename):
    """Return true if the file is compilable (should be in OBJS)."""
    for res in (filename.endswith(e) for e in COMPILABLE_EXTENSIONS):
        if res:
            return True
    return False