import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def Target(filename):
    """Translate a compilable filename to its .o target."""
    return os.path.splitext(filename)[0] + '.o'