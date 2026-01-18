import errno
import os
import sys
import tempfile
from subprocess import PIPE, Popen
from .errors import BzrError, NoDiff3
from .textfile import check_text_path
class PatchInvokeError(BzrError):
    _fmt = 'Error invoking patch: %(errstr)s%(stderr)s'
    internal_error = False

    def __init__(self, e, stderr=''):
        self.exception = e
        self.errstr = os.strerror(e.errno)
        self.stderr = '\n' + stderr