import bisect
import errno
import itertools
import os
import random
import stat
import sys
import time
import warnings
from .. import config, debug, errors, urlutils
from ..errors import LockError, ParamikoNotPresent, PathError, TransportError
from ..osutils import fancy_rename
from ..trace import mutter, warning
from ..transport import (ConnectedTransport, FileExists, FileFileStream,
def _b_for_broken_paramiko(s, encoding='utf8'):
    """Hacked b() that does not raise TypeError."""
    if not isinstance(s, bytes):
        encode = getattr(s, 'encode', None)
        if encode is not None:
            return encode(encoding)
        tobytes = getattr(s, 'tobytes', None)
        if tobytes is not None:
            return tobytes()
    return s