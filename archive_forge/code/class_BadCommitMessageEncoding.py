import codecs
import os
import sys
from io import BytesIO, StringIO
from subprocess import call
from . import bedding, cmdline, config, osutils, trace, transport, ui
from .errors import BzrError
from .hooks import Hooks
class BadCommitMessageEncoding(BzrError):
    _fmt = 'The specified commit message contains characters unsupported by the current encoding.'