import os
import platform
import subprocess
import sys
from ._version import version as __version__
def _program(name, args):
    return subprocess.call([os.path.join(BIN_DIR, name)] + args, close_fds=False)