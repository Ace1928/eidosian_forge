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
def _populate_staging(self, name=None, version=None, static_url=None, clean=False):
    """Set up the assets in the staging directory."""
    app_dir = self.app_dir
    staging = pjoin(app_dir, 'staging')
    if clean and osp.exists(staging):
        self.logger.info('Cleaning %s', staging)
        _rmtree(staging, self.logger)
    self._ensure_app_dirs()
    if not version:
        version = self.info['core_data']['jupyterlab']['version']
    splice_source = self._options.splice_source
    if splice_source:
        self.logger.debug('Splicing dev packages into app directory.')
        source_dir = DEV_DIR
        version = __version__ + '-spliced'
    else:
        source_dir = pjoin(HERE, 'staging')
    pkg_path = pjoin(staging, 'package.json')
    if osp.exists(pkg_path):
        with open(pkg_path) as fid:
            data = json.load(fid)
        if data['jupyterlab'].get('version', '') != version:
            _rmtree(staging, self.logger)
            os.makedirs(staging)
    for fname in ['index.js', 'bootstrap.js', 'publicpath.js', 'webpack.config.js', 'webpack.prod.config.js', 'webpack.prod.minimize.config.js']:
        target = pjoin(staging, fname)
        shutil.copy(pjoin(source_dir, fname), target)
    for fname in ['.yarnrc.yml', 'yarn.js']:
        target = pjoin(staging, fname)
        shutil.copy(pjoin(HERE, 'staging', fname), target)
    templates = pjoin(staging, 'templates')
    if osp.exists(templates):
        _rmtree(templates, self.logger)
    try:
        shutil.copytree(pjoin(source_dir, 'templates'), templates)
    except shutil.Error as error:
        real_error = '[Errno 22]' not in str(error) and '[Errno 5]' not in str(error)
        if real_error or not osp.exists(templates):
            raise
    linked_dir = pjoin(staging, 'linked_packages')
    if osp.exists(linked_dir):
        _rmtree(linked_dir, self.logger)
    os.makedirs(linked_dir)
    extensions = self.info['extensions']
    removed = False
    for key, source in self.info['local_extensions'].items():
        if key not in extensions:
            config = self._read_build_config()
            data = config.setdefault('local_extensions', {})
            del data[key]
            self._write_build_config(config)
            removed = True
            continue
        dname = pjoin(app_dir, 'extensions')
        self._update_local(key, source, dname, extensions[key], 'local_extensions')
    if removed:
        self.info['local_extensions'] = self._get_local_extensions()
    linked = self.info['linked_packages']
    for key, item in linked.items():
        dname = pjoin(staging, 'linked_packages')
        self._update_local(key, item['source'], dname, item, 'linked_packages')
    data = self._get_package_template()
    jlab = data['jupyterlab']
    if version:
        jlab['version'] = version
    if name:
        jlab['name'] = name
    if static_url:
        jlab['staticUrl'] = static_url
    if splice_source:
        for path in glob(pjoin(REPO_ROOT, 'packages', '*', 'package.json')):
            local_path = osp.dirname(osp.abspath(path))
            pkg_data = json.loads(Path(path).read_text(encoding='utf-8'))
            name = pkg_data['name']
            if name in data['dependencies']:
                data['dependencies'][name] = local_path
                jlab['linkedPackages'][name] = local_path
            if name in data['resolutions']:
                data['resolutions'][name] = local_path
        local_path = osp.abspath(pjoin(REPO_ROOT, 'builder'))
        data['devDependencies']['@jupyterlab/builder'] = local_path
        target = osp.join(staging, 'node_modules', '@jupyterlab', 'builder')
        node_modules = pjoin(staging, 'node_modules')
        if osp.exists(node_modules):
            shutil.rmtree(node_modules, ignore_errors=True)
    pkg_path = pjoin(staging, 'package.json')
    with open(pkg_path, 'w') as fid:
        json.dump(data, fid, indent=4)
    lock_path = pjoin(staging, 'yarn.lock')
    lock_template = pjoin(HERE, 'staging', 'yarn.lock')
    if not osp.exists(lock_path):
        shutil.copy(lock_template, lock_path)
        os.chmod(lock_path, stat.S_IWRITE | stat.S_IREAD)