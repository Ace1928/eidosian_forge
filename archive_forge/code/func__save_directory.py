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
def _save_directory(self, os_path, model, path=''):
    """create a directory"""
    if not self.allow_hidden and is_hidden(os_path, self.root_dir):
        raise web.HTTPError(400, 'Cannot create directory %r' % os_path)
    if not os.path.exists(os_path):
        with self.perm_to_403():
            os.mkdir(os_path)
    elif not os.path.isdir(os_path):
        raise web.HTTPError(400, 'Not a directory: %s' % os_path)
    else:
        self.log.debug('Directory %r already exists', os_path)