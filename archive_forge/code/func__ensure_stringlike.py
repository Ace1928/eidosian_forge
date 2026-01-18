import sys
import os
import re
import logging
from .errors import DistutilsOptionError
from . import util, dir_util, file_util, archive_util, _modified
from ._log import log
def _ensure_stringlike(self, option, what, default=None):
    val = getattr(self, option)
    if val is None:
        setattr(self, option, default)
        return default
    elif not isinstance(val, str):
        raise DistutilsOptionError("'{}' must be a {} (got `{}`)".format(option, what, val))
    return val