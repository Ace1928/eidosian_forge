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
def _get_package_template(self, silent=False):
    """Get the template the for staging package.json file."""
    logger = self.logger
    data = deepcopy(self.info['core_data'])
    local = self.info['local_extensions']
    linked = self.info['linked_packages']
    extensions = self.info['extensions']
    shadowed_exts = self.info['shadowed_exts']
    jlab = data['jupyterlab']

    def format_path(path):
        path = osp.relpath(path, osp.abspath(osp.realpath(pjoin(self.app_dir, 'staging'))))
        path = 'file:' + path.replace(os.sep, '/')
        if os.name == 'nt':
            path = path.lower()
        return path
    jlab['linkedPackages'] = {}
    for key, source in local.items():
        if key in shadowed_exts:
            continue
        jlab['linkedPackages'][key] = source
        data['resolutions'][key] = 'file:' + self.info['extensions'][key]['path']
    for key, item in linked.items():
        if key in shadowed_exts:
            continue
        path = pjoin(self.app_dir, 'staging', 'linked_packages')
        path = pjoin(path, item['filename'])
        data['dependencies'][key] = format_path(path)
        jlab['linkedPackages'][key] = item['source']
        data['resolutions'][key] = format_path(path)
    data['jupyterlab']['extensionMetadata'] = {}
    compat_errors = self._get_extension_compat()
    for key, value in extensions.items():
        errors = compat_errors[key]
        if errors:
            if not silent:
                _log_single_compat_errors(logger, key, value['version'], errors)
            continue
        data['dependencies'][key] = format_path(value['path'])
        jlab_data = value['jupyterlab']
        for item in ['extension', 'mimeExtension']:
            ext = jlab_data.get(item, False)
            if not ext:
                continue
            if ext is True:
                ext = ''
            jlab[item + 's'][key] = ext
            data['jupyterlab']['extensionMetadata'][key] = jlab_data
    for item in self.info['uninstalled_core']:
        if item in jlab['extensions']:
            data['jupyterlab']['extensions'].pop(item)
        elif item in jlab['mimeExtensions']:
            data['jupyterlab']['mimeExtensions'].pop(item)
        if item in data['dependencies']:
            data['dependencies'].pop(item)
    return data