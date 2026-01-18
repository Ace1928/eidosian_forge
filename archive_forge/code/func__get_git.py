from __future__ import annotations
from .. import mlog
import contextlib
from dataclasses import dataclass
import urllib.request
import urllib.error
import urllib.parse
import os
import hashlib
import shutil
import tempfile
import stat
import subprocess
import sys
import configparser
import time
import typing as T
import textwrap
import json
from base64 import b64encode
from netrc import netrc
from pathlib import Path, PurePath
from functools import lru_cache
from . import WrapMode
from .. import coredata
from ..mesonlib import quiet_git, GIT, ProgressBar, MesonException, windows_proof_rmtree, Popen_safe
from ..interpreterbase import FeatureNew
from ..interpreterbase import SubProject
from .. import mesonlib
def _get_git(self, packagename: str) -> None:
    if not GIT:
        raise WrapException(f'Git program not found, cannot download {packagename}.wrap via git.')
    revno = self.wrap.get('revision')
    checkout_cmd = ['-c', 'advice.detachedHead=false', 'checkout', revno, '--']
    is_shallow = False
    depth_option: T.List[str] = []
    if self.wrap.values.get('depth', '') != '':
        is_shallow = True
        depth_option = ['--depth', self.wrap.values.get('depth')]
    if is_shallow and self.is_git_full_commit_id(revno):
        verbose_git(['-c', 'init.defaultBranch=meson-dummy-branch', 'init', self.directory], self.subdir_root, check=True)
        verbose_git(['remote', 'add', 'origin', self.wrap.get('url')], self.dirname, check=True)
        revno = self.wrap.get('revision')
        verbose_git(['fetch', *depth_option, 'origin', revno], self.dirname, check=True)
        verbose_git(checkout_cmd, self.dirname, check=True)
    elif not is_shallow:
        verbose_git(['clone', self.wrap.get('url'), self.directory], self.subdir_root, check=True)
        if revno.lower() != 'head':
            if not verbose_git(checkout_cmd, self.dirname):
                verbose_git(['fetch', self.wrap.get('url'), revno], self.dirname, check=True)
                verbose_git(checkout_cmd, self.dirname, check=True)
    else:
        args = ['-c', 'advice.detachedHead=false', 'clone', *depth_option]
        if revno.lower() != 'head':
            args += ['--branch', revno]
        args += [self.wrap.get('url'), self.directory]
        verbose_git(args, self.subdir_root, check=True)
    if self.wrap.values.get('clone-recursive', '').lower() == 'true':
        verbose_git(['submodule', 'update', '--init', '--checkout', '--recursive', *depth_option], self.dirname, check=True)
    push_url = self.wrap.values.get('push-url')
    if push_url:
        verbose_git(['remote', 'set-url', '--push', 'origin', push_url], self.dirname, check=True)