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
def _semver_prerelease_key(prerelease):
    """Sort key for prereleases.

    Precedence for two pre-release versions with the same
    major, minor, and patch version MUST be determined by
    comparing each dot separated identifier from left to
    right until a difference is found as follows:
    identifiers consisting of only digits are compare
    numerically and identifiers with letters or hyphens
    are compared lexically in ASCII sort order. Numeric
    identifiers always have lower precedence than non-
    numeric identifiers. A larger set of pre-release
    fields has a higher precedence than a smaller set,
    if all of the preceding identifiers are equal.
    """
    for entry in prerelease:
        if isinstance(entry, int):
            yield ('', entry)
        else:
            yield (entry,)