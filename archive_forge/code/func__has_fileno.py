import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
def _has_fileno(f):
    """ test that a file-like object is really a filehandle

    Only filehandles can be given to apt_pkg.TagFile.
    """
    try:
        f.fileno()
        return True
    except (AttributeError, io.UnsupportedOperation):
        return False