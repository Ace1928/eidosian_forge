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
def _semver_key(version, prerelease_first=False):
    """A sort key-function for sorting semver version string.

    The default sorting order is ascending (0.x -> 1.x -> 2.x).

    If `prerelease_first`, pre-releases will come before
    ALL other semver keys (not just those with same version).
    I.e (1.0-pre, 2.0-pre -> 0.x -> 1.x -> 2.x).

    Otherwise it will sort in the standard way that it simply
    comes before any release with shared version string
    (0.x -> 1.0-pre -> 1.x -> 2.0-pre -> 2.x).
    """
    v = make_semver(version, True)
    key = ((0,) if v.prerelease else (1,)) if prerelease_first else ()
    key = (*key, v.major, v.minor, v.patch)
    if not prerelease_first:
        key = (*key, 0) if v.prerelease else (1,)
    if v.prerelease:
        key = key + tuple(_semver_prerelease_key(v.prerelease))
    return key