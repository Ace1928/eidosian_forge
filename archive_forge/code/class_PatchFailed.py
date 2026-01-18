import errno
import os
import sys
import tempfile
from subprocess import PIPE, Popen
from .errors import BzrError, NoDiff3
from .textfile import check_text_path
class PatchFailed(BzrError):
    _fmt = 'Patch application failed'