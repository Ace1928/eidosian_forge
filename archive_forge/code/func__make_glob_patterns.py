import datetime
import decimal
import glob
import numbers
import os
import shutil
import string
from functools import partial
from stat import ST_DEV, ST_INO
from . import _string_parsers as string_parsers
from ._ctime_functions import get_ctime, set_ctime
from ._datetime import aware_now
@staticmethod
def _make_glob_patterns(path):
    formatter = string.Formatter()
    tokens = formatter.parse(path)
    escaped = ''.join((glob.escape(text) + '*' * (name is not None) for text, name, *_ in tokens))
    root, ext = os.path.splitext(escaped)
    if not ext:
        return [escaped, escaped + '.*']
    return [escaped, escaped + '.*', root + '.*' + ext, root + '.*' + ext + '.*']