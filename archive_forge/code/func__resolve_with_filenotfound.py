import sys
import pickle
import errno
import subprocess as sp
import gzip
import hashlib
import locale
from hashlib import md5
import os
import os.path as op
import re
import shutil
import contextlib
import posixpath
from pathlib import Path
import simplejson as json
from time import sleep, time
from .. import logging, config, __version__ as version
from .misc import is_container
def _resolve_with_filenotfound(path, **kwargs):
    """Raise FileNotFoundError instead of OSError"""
    try:
        return path.resolve(**kwargs)
    except OSError as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise FileNotFoundError(str(path))