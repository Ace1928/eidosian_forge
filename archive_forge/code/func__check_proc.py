import collections
import subprocess
import warnings
from . import protocols
from . import transports
from .log import logger
def _check_proc(self):
    if self._proc is None:
        raise ProcessLookupError()