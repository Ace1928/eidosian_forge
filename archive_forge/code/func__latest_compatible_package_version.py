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
def _latest_compatible_package_version(self, name):
    """Get the latest compatible version of a package"""
    core_data = self.info['core_data']
    try:
        metadata = _fetch_package_metadata(self.registry, name, self.logger)
    except URLError:
        return
    versions = metadata.get('versions', {})

    def sort_key(key_value):
        return _semver_key(key_value[0], prerelease_first=True)
    for version, data in sorted(versions.items(), key=sort_key, reverse=True):
        deps = data.get('dependencies', {})
        errors = _validate_compatibility(name, deps, core_data)
        if not errors:
            if 'deprecated' in data:
                self.logger.debug(f'Disregarding compatible version of package as it is deprecated: {name}@{version}')
                continue
            with TemporaryDirectory() as tempdir:
                info = self._extract_package(f'{name}@{version}', tempdir)
            if _validate_extension(info['data']):
                return
            return version