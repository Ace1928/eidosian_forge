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
def _is_locked(name, locked=None) -> LockStatus:
    """Test whether the package is locked.

    If only a subset of extension plugins is locked return them.
    """
    locked = locked or {}
    locked_plugins = set()
    for lock, value in locked.items():
        if value is False:
            continue
        if name == lock:
            return LockStatus(entire_extension_locked=True)
        extension_part = lock.partition(':')[0]
        if name == extension_part:
            locked_plugins.add(lock)
    return LockStatus(entire_extension_locked=False, locked_plugins=locked_plugins)