import fcntl
import fnmatch
import glob
import json
import os
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
def ExecFlock(self, lockfile, *cmd_list):
    """Emulates the most basic behavior of Linux's flock(1)."""
    fd = os.open(lockfile, os.O_RDONLY | os.O_NOCTTY | os.O_CREAT, 438)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return subprocess.call(cmd_list)