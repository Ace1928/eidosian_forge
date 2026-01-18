import re
from threading import RLock
from typing import Any, Dict, Tuple
from urllib.parse import urlparse
from triad.utils.hash import to_uuid
import fs
from fs import memoryfs, open_fs, tempfs
from fs.base import FS as FSBase
from fs.glob import BoundGlobber, Globber
from fs.mountfs import MountFS
from fs.subfs import SubFS
Make a directory, and any missing intermediate directories.

        .. note::

            This overrides the base ``makedirs``

        :param path: path to directory from root.
        :param permissions: initial permissions, or `None` to use defaults.
        :recreate: if `False` (the default), attempting to
          create an existing directory will raise an error. Set
          to `True` to ignore existing directories.
        :return: a sub-directory filesystem.

        :raises fs.errors.DirectoryExists: if the path is already
          a directory, and ``recreate`` is `False`.
        :raises fs.errors.DirectoryExpected: if one of the ancestors
          in the path is not a directory.
        