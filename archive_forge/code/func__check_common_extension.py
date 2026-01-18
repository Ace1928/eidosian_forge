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
def _check_common_extension(self, extension, info, check_installed_only):
    """Check if a common (non-core) extension is enabled or disabled"""
    if extension not in info['extensions']:
        self.logger.info(f'{extension}:{RED_X}')
        return False
    errors = self._get_extension_compat()[extension]
    if errors:
        self.logger.info(f'{extension}:{RED_X} (compatibility errors)')
        return False
    if check_installed_only:
        self.logger.info(f'{extension}: {GREEN_OK}')
        return True
    if _is_disabled(extension, info['disabled']):
        self.logger.info(f'{extension}: {RED_DISABLED}')
        return False
    self.logger.info(f'{extension}:{GREEN_ENABLED}')
    return True