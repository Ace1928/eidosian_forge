import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
def _detect_pathlib_path(p):
    if (3, 4) <= sys.version_info:
        import pathlib
        if isinstance(p, pathlib.PurePath):
            return True
    return False