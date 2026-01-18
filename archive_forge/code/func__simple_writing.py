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
@contextmanager
def _simple_writing(path, text=True, encoding='utf-8', log=None, **kwargs):
    """Context manager to write file without doing atomic writing
    (for weird filesystem eg: nfs).

    Parameters
    ----------
    path : str
        The target file to write to.
    text : bool, optional
        Whether to open the file in text mode (i.e. to write unicode). Default is
        True.
    encoding : str, optional
        The encoding to use for files opened in text mode. Default is UTF-8.
    **kwargs
        Passed to :func:`io.open`.
    """
    if os.path.islink(path):
        path = os.path.join(os.path.dirname(path), os.readlink(path))
    if text:
        kwargs.setdefault('newline', '\n')
        fileobj = open(path, 'w', encoding=encoding, **kwargs)
    else:
        fileobj = open(path, 'wb', **kwargs)
    try:
        yield fileobj
    except BaseException:
        fileobj.close()
        raise
    fileobj.close()