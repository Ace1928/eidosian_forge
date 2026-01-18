import importlib
import json
import os
import os.path as osp
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from os.path import basename, normpath
from os.path import join as pjoin
from jupyter_core.paths import ENV_JUPYTER_PATH, SYSTEM_JUPYTER_PATH, jupyter_data_dir
from jupyter_core.utils import ensure_dir_exists
from jupyter_server.extension.serverextension import ArgumentConflict
from jupyterlab_server.config import get_federated_extensions
from .commands import _test_overlap
def _ensure_builder(ext_path, core_path):
    """Ensure that we can build the extension and return the builder script path"""
    with open(osp.join(core_path, 'package.json')) as fid:
        core_data = json.load(fid)
    with open(osp.join(ext_path, 'package.json')) as fid:
        ext_data = json.load(fid)
    dep_version1 = core_data['devDependencies']['@jupyterlab/builder']
    dep_version2 = ext_data.get('devDependencies', {}).get('@jupyterlab/builder')
    dep_version2 = dep_version2 or ext_data.get('dependencies', {}).get('@jupyterlab/builder')
    if dep_version2 is None:
        raise ValueError('Extensions require a devDependency on @jupyterlab/builder@%s' % dep_version1)
    if '/' in dep_version2:
        with open(osp.join(ext_path, dep_version2, 'package.json')) as fid:
            dep_version2 = json.load(fid).get('version')
    if not osp.exists(osp.join(ext_path, 'node_modules')):
        subprocess.check_call(['jlpm'], cwd=ext_path)
    target = ext_path
    while not osp.exists(osp.join(target, 'node_modules', '@jupyterlab', 'builder')):
        if osp.dirname(target) == target:
            msg = 'Could not find @jupyterlab/builder'
            raise ValueError(msg)
        target = osp.dirname(target)
    overlap = _test_overlap(dep_version1, dep_version2, drop_prerelease1=True, drop_prerelease2=True)
    if not overlap:
        with open(osp.join(target, 'node_modules', '@jupyterlab', 'builder', 'package.json')) as fid:
            dep_version2 = json.load(fid).get('version')
        overlap = _test_overlap(dep_version1, dep_version2, drop_prerelease1=True, drop_prerelease2=True)
    if not overlap:
        msg = f'Extensions require a devDependency on @jupyterlab/builder@{dep_version1}, you have a dependency on {dep_version2}'
        raise ValueError(msg)
    return osp.join(target, 'node_modules', '@jupyterlab', 'builder', 'lib', 'build-labextension.js')