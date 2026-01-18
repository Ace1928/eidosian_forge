import functools
import itertools
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import warnings
import zipfile
from hashlib import md5
from xml.etree import ElementTree
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import nltk
def _table_reprfunc(self, row, col, val):
    if self._table.column_names[col].endswith('Size'):
        if isinstance(val, str):
            return '  %s' % val
        elif val < 1024 ** 2:
            return '  %.1f KB' % (val / 1024.0 ** 1)
        elif val < 1024 ** 3:
            return '  %.1f MB' % (val / 1024.0 ** 2)
        else:
            return '  %.1f GB' % (val / 1024.0 ** 3)
    if col in (0, ''):
        return str(val)
    else:
        return '  %s' % val