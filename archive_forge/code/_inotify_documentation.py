import ctypes
import ctypes.util
from typing import Any, cast
from twisted.python.filepath import FilePath

    Initialize the module, checking if the expected APIs exist and setting the
    argtypes and restype for C{inotify_init}, C{inotify_add_watch}, and
    C{inotify_rm_watch}.
    