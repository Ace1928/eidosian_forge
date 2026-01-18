import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def UnrelativePath(path, relative_to):
    rel_dir = os.path.dirname(relative_to)
    return os.path.normpath(os.path.join(rel_dir, path))