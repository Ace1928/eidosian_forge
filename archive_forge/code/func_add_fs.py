from __future__ import absolute_import, print_function, unicode_literals
import typing
from collections import OrderedDict, namedtuple
from operator import itemgetter
from six import text_type
from . import errors
from .base import FS
from .mode import check_writable
from .opener import open_fs
from .path import abspath, normpath
def add_fs(self, name, fs, write=False, priority=0):
    """Add a filesystem to the MultiFS.

        Arguments:
            name (str): A unique name to refer to the filesystem being
                added.
            fs (FS or str): The filesystem (instance or URL) to add.
            write (bool): If this value is True, then the ``fs`` will
                be used as the writeable FS (defaults to False).
            priority (int): An integer that denotes the priority of the
                filesystem being added. Filesystems will be searched in
                descending priority order and then by the reverse order
                they were added. So by default, the most recently added
                filesystem will be looked at first.

        """
    if isinstance(fs, text_type):
        fs = open_fs(fs)
    if not isinstance(fs, FS):
        raise TypeError('fs argument should be an FS object or FS URL')
    self._filesystems[name] = _PrioritizedFS(priority=(priority, self._sort_index), fs=fs)
    self._sort_index += 1
    self._resort()
    if write:
        self.write_fs = fs
        self._write_fs_name = name