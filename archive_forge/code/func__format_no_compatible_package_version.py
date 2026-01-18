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
def _format_no_compatible_package_version(self, name):
    """Get the latest compatible version of a package"""
    core_data = self.info['core_data']
    lab_newer_than_latest = False
    latest_newer_than_lab = False
    try:
        metadata = _fetch_package_metadata(self.registry, name, self.logger)
    except URLError:
        pass
    else:
        versions = metadata.get('versions', {})

        def sort_key(key_value):
            return _semver_key(key_value[0], prerelease_first=True)
        store = tuple(sorted(versions.items(), key=sort_key, reverse=True))
        latest_deps = store[0][1].get('dependencies', {})
        core_deps = core_data['resolutions']
        singletons = core_data['jupyterlab']['singletonPackages']
        for key, value in latest_deps.items():
            if key in singletons:
                c = _compare_ranges(core_deps[key], value, drop_prerelease1=True)
                lab_newer_than_latest = lab_newer_than_latest or c < 0
                latest_newer_than_lab = latest_newer_than_lab or c > 0
    if lab_newer_than_latest:
        return 'The extension "%s" does not yet support the current version of JupyterLab.\n' % name
    parts = ['No version of {extension} could be found that is compatible with the current version of JupyterLab.']
    if latest_newer_than_lab:
        parts.extend(('However, it seems to support a new version of JupyterLab.', 'Consider upgrading JupyterLab.'))
    return ' '.join(parts).format(extension=name)