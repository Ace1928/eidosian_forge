import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
def _ispath(p):
    if isinstance(p, (bytes, basestring)):
        return True
    return _detect_pathlib_path(p)