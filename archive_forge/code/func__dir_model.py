from __future__ import annotations
import errno
import math
import mimetypes
import os
import platform
import shutil
import stat
import subprocess
import sys
import typing as t
import warnings
from datetime import datetime
from pathlib import Path
import nbformat
from anyio.to_thread import run_sync
from jupyter_core.paths import exists, is_file_hidden, is_hidden
from send2trash import send2trash
from tornado import web
from traitlets import Bool, Int, TraitError, Unicode, default, validate
from jupyter_server import _tz as tz
from jupyter_server.base.handlers import AuthenticatedFileHandler
from jupyter_server.transutils import _i18n
from jupyter_server.utils import to_api_path
from .filecheckpoints import AsyncFileCheckpoints, FileCheckpoints
from .fileio import AsyncFileManagerMixin, FileManagerMixin
from .manager import AsyncContentsManager, ContentsManager, copy_pat
def _dir_model(self, path, content=True):
    """Build a model for a directory

        if content is requested, will include a listing of the directory
        """
    os_path = self._get_os_path(path)
    four_o_four = 'directory does not exist: %r' % path
    if not os.path.isdir(os_path):
        raise web.HTTPError(404, four_o_four)
    elif not self.allow_hidden and is_hidden(os_path, self.root_dir):
        self.log.info('Refusing to serve hidden directory %r, via 404 Error', os_path)
        raise web.HTTPError(404, four_o_four)
    model = self._base_model(path)
    model['type'] = 'directory'
    model['size'] = None
    if content:
        model['content'] = contents = []
        os_dir = self._get_os_path(path)
        for name in os.listdir(os_dir):
            try:
                os_path = os.path.join(os_dir, name)
            except UnicodeDecodeError as e:
                self.log.warning("failed to decode filename '%s': %r", name, e)
                continue
            try:
                st = os.lstat(os_path)
            except OSError as e:
                if e.errno == errno.ENOENT:
                    self.log.warning("%s doesn't exist", os_path)
                elif e.errno != errno.EACCES:
                    self.log.warning('Error stat-ing %s: %r', os_path, e)
                continue
            if not stat.S_ISLNK(st.st_mode) and (not stat.S_ISREG(st.st_mode)) and (not stat.S_ISDIR(st.st_mode)):
                self.log.debug('%s not a regular file', os_path)
                continue
            try:
                if self.should_list(name) and (self.allow_hidden or not is_file_hidden(os_path, stat_res=st)):
                    contents.append(self.get(path=f'{path}/{name}', content=False))
            except OSError as e:
                if e.errno not in [errno.ELOOP, errno.EACCES]:
                    self.log.warning('Unknown error checking if file %r is hidden', os_path, exc_info=True)
        model['format'] = 'json'
    return model