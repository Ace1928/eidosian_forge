import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
@property
def __version__(self):
    version_info = self.version_info()
    if version_info is None or len(version_info) == 0:
        return 'unknown'
    try:
        version_string = breezy._format_version_tuple(version_info)
    except (ValueError, TypeError, IndexError):
        trace.log_exception_quietly()
        version_string = '.'.join(map(str, version_info))
    return version_string