from __future__ import annotations
import errno
import hashlib
import os
import shutil
from base64 import decodebytes, encodebytes
from contextlib import contextmanager
from functools import partial
import nbformat
from anyio.to_thread import run_sync
from tornado.web import HTTPError
from traitlets import Bool, Enum
from traitlets.config import Configurable
from traitlets.config.configurable import LoggingConfigurable
from jupyter_server.utils import ApiPath, to_api_path, to_os_path
def _get_os_path(self, path):
    """Given an API path, return its file system path.

        Parameters
        ----------
        path : str
            The relative API path to the named file.

        Returns
        -------
        path : str
            Native, absolute OS path to for a file.

        Raises
        ------
        404: if path is outside root
        """
    self.log.debug('Reading path from disk: %s', path)
    root = os.path.abspath(self.root_dir)
    if os.path.splitdrive(path)[0]:
        raise HTTPError(404, '%s is not a relative API path' % path)
    os_path = to_os_path(ApiPath(path), root)
    try:
        os.lstat(os_path)
    except OSError:
        pass
    except ValueError:
        raise HTTPError(404, f'{path} is not a valid path') from None
    if not (os.path.abspath(os_path) + os.path.sep).startswith(root):
        raise HTTPError(404, '%s is outside root contents directory' % path)
    return os_path