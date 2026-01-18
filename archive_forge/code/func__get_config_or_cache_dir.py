import atexit
from collections import namedtuple
from collections.abc import MutableMapping
import contextlib
import functools
import importlib
import inspect
from inspect import Parameter
import locale
import logging
import os
from pathlib import Path
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
import numpy
from packaging.version import parse as parse_version
from . import _api, _version, cbook, _docstring, rcsetup
from matplotlib.cbook import sanitize_sequence
from matplotlib._api import MatplotlibDeprecationWarning
from matplotlib.rcsetup import validate_backend, cycler
from matplotlib.cm import _colormaps as colormaps
from matplotlib.colors import _color_sequences as color_sequences
def _get_config_or_cache_dir(xdg_base_getter):
    configdir = os.environ.get('MPLCONFIGDIR')
    if configdir:
        configdir = Path(configdir).resolve()
    elif sys.platform.startswith(('linux', 'freebsd')):
        configdir = Path(xdg_base_getter(), 'matplotlib')
    else:
        configdir = Path.home() / '.matplotlib'
    try:
        configdir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    else:
        if os.access(str(configdir), os.W_OK) and configdir.is_dir():
            return str(configdir)
    try:
        tmpdir = tempfile.mkdtemp(prefix='matplotlib-')
    except OSError as exc:
        raise OSError(f'Matplotlib requires access to a writable cache directory, but the default path ({configdir}) is not a writable directory, and a temporary directory could not be created; set the MPLCONFIGDIR environment variable to a writable directory') from exc
    os.environ['MPLCONFIGDIR'] = tmpdir
    atexit.register(shutil.rmtree, tmpdir)
    _log.warning('Matplotlib created a temporary cache directory at %s because the default path (%s) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.', tmpdir, configdir)
    return tmpdir