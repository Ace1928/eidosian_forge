import contextlib
import errno
import hashlib
import itertools
import json
import logging
import os
import os.path as osp
import re
import shutil
import site
import stat
import subprocess
import sys
import tarfile
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import FrozenSet, Optional
from urllib.error import URLError
from urllib.request import Request, quote, urljoin, urlopen
from jupyter_core.paths import jupyter_config_dir
from jupyter_server.extension.serverextension import GREEN_ENABLED, GREEN_OK, RED_DISABLED, RED_X
from jupyterlab_server.config import (
from jupyterlab_server.process import Process, WatchHelper, list2cmdline, which
from packaging.version import Version
from traitlets import Bool, HasTraits, Instance, List, Unicode, default
from jupyterlab._version import __version__
from jupyterlab.coreconfig import CoreConfig
from jupyterlab.jlpmapp import HERE, YARN_PATH
from jupyterlab.semver import Range, gt, gte, lt, lte, make_semver
def _format_compatibility_errors(name, version, errors):
    """Format a message for compatibility errors."""
    msgs = []
    l0 = 10
    l1 = 10
    for error in errors:
        pkg, jlab, ext = error
        jlab = str(Range(jlab, True))
        ext = str(Range(ext, True))
        msgs.append((pkg, jlab, ext))
        l0 = max(l0, len(pkg) + 1)
        l1 = max(l1, len(jlab) + 1)
    msg = '\n"%s@%s" is not compatible with the current JupyterLab'
    msg = msg % (name, version)
    msg += '\nConflicting Dependencies:\n'
    msg += 'JupyterLab'.ljust(l0)
    msg += 'Extension'.ljust(l1)
    msg += 'Package\n'
    for pkg, jlab, ext in msgs:
        msg += jlab.ljust(l0) + ext.ljust(l1) + pkg + '\n'
    return msg