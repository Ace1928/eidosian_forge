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
def _rc_params_in_file(fname, transform=lambda x: x, fail_on_error=False):
    """
    Construct a `RcParams` instance from file *fname*.

    Unlike `rc_params_from_file`, the configuration class only contains the
    parameters specified in the file (i.e. default values are not filled in).

    Parameters
    ----------
    fname : path-like
        The loaded file.
    transform : callable, default: the identity function
        A function called on each individual line of the file to transform it,
        before further parsing.
    fail_on_error : bool, default: False
        Whether invalid entries should result in an exception or a warning.
    """
    import matplotlib as mpl
    rc_temp = {}
    with _open_file_or_url(fname) as fd:
        try:
            for line_no, line in enumerate(fd, 1):
                line = transform(line)
                strippedline = cbook._strip_comment(line)
                if not strippedline:
                    continue
                tup = strippedline.split(':', 1)
                if len(tup) != 2:
                    _log.warning('Missing colon in file %r, line %d (%r)', fname, line_no, line.rstrip('\n'))
                    continue
                key, val = tup
                key = key.strip()
                val = val.strip()
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                if key in rc_temp:
                    _log.warning('Duplicate key in file %r, line %d (%r)', fname, line_no, line.rstrip('\n'))
                rc_temp[key] = (val, line, line_no)
        except UnicodeDecodeError:
            _log.warning('Cannot decode configuration file %r as utf-8.', fname)
            raise
    config = RcParams()
    for key, (val, line, line_no) in rc_temp.items():
        if key in rcsetup._validators:
            if fail_on_error:
                config[key] = val
            else:
                try:
                    config[key] = val
                except Exception as msg:
                    _log.warning('Bad value in file %r, line %d (%r): %s', fname, line_no, line.rstrip('\n'), msg)
        elif key in _deprecated_ignore_map:
            version, alt_key = _deprecated_ignore_map[key]
            _api.warn_deprecated(version, name=key, alternative=alt_key, obj_type='rcparam', addendum='Please update your matplotlibrc.')
        else:
            version = 'main' if '.post' in mpl.__version__ else f'v{mpl.__version__}'
            _log.warning('\nBad key %(key)s in file %(fname)s, line %(line_no)s (%(line)r)\nYou probably need to get an updated matplotlibrc file from\nhttps://github.com/matplotlib/matplotlib/blob/%(version)s/lib/matplotlib/mpl-data/matplotlibrc\nor from the matplotlib source distribution', dict(key=key, fname=fname, line_no=line_no, line=line.rstrip('\n'), version=version))
    return config