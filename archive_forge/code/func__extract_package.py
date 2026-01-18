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
def _extract_package(self, source, tempdir, pin=None):
    """Call `npm pack` for an extension.

        The pack command will download the package tar if `source` is
        a package name, or run `npm pack` locally if `source` is a
        directory.
        """
    is_dir = osp.exists(source) and osp.isdir(source)
    if is_dir and (not osp.exists(pjoin(source, 'node_modules'))):
        self._run(['node', YARN_PATH, 'install'], cwd=source)
    info = {'source': source, 'is_dir': is_dir}
    ret = self._run([which('npm'), 'pack', source], cwd=tempdir)
    if ret != 0:
        msg = '"%s" is not a valid npm package'
        raise ValueError(msg % source)
    path = glob(pjoin(tempdir, '*.tgz'))[0]
    info['data'] = read_package(path)
    if is_dir:
        info['sha'] = sha = _tarsum(path)
        target = path.replace('.tgz', '-%s.tgz' % sha)
        shutil.move(path, target)
        info['path'] = target
    else:
        info['path'] = path
    if pin:
        old_path = info['path']
        new_path = pjoin(osp.dirname(old_path), f'{PIN_PREFIX}{pin}.tgz')
        shutil.move(old_path, new_path)
        info['path'] = new_path
    info['filename'] = osp.basename(info['path'])
    info['name'] = info['data']['name']
    info['version'] = info['data']['version']
    return info