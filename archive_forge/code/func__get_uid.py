import os
from warnings import warn
import sys
from distutils.errors import DistutilsExecError
from distutils.spawn import spawn
from distutils.dir_util import mkpath
from distutils import log
def _get_uid(name):
    """Returns an uid, given a user name."""
    if getpwnam is None or name is None:
        return None
    try:
        result = getpwnam(name)
    except KeyError:
        result = None
    if result is not None:
        return result[2]
    return None