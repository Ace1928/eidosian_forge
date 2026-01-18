import os
import shutil
import subprocess
import sys
import tempfile
from .lazy_import import lazy_import
from breezy import (
def _get_executable_path(exe):
    if os.path.isabs(exe):
        return exe
    return osutils.find_executable_on_path(exe)