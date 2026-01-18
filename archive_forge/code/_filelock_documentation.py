import os
from filelock import FileLock as FileLock_
from filelock import UnixFileLock
from filelock import __version__ as _filelock_version
from packaging import version

    A `filelock.FileLock` initializer that handles long paths.
    It also uses the current umask for lock files.
    