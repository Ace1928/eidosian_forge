import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def FixIfRelativePath(path, relative_to):
    if os.path.isabs(path):
        return path
    return RelativePath(path, relative_to)