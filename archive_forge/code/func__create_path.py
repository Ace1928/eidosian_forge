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
def _create_path(self):
    path = self._path.format_map({'time': FileDateFormatter()})
    return os.path.abspath(path)