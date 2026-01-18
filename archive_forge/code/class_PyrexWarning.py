from __future__ import absolute_import
import sys
from contextlib import contextmanager
from ..Utils import open_new_file
from . import DebugFlags
from . import Options
class PyrexWarning(Exception):
    pass