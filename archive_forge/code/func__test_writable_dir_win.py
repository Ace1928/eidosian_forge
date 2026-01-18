import fnmatch
import os
import os.path
import random
import sys
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO, Generator, List, Union, cast
from pip._vendor.tenacity import retry, stop_after_delay, wait_fixed
from pip._internal.utils.compat import get_path_uid
from pip._internal.utils.misc import format_size
def _test_writable_dir_win(path: str) -> bool:
    basename = 'accesstest_deleteme_fishfingers_custard_'
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
    for _ in range(10):
        name = basename + ''.join((random.choice(alphabet) for _ in range(6)))
        file = os.path.join(path, name)
        try:
            fd = os.open(file, os.O_RDWR | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            pass
        except PermissionError:
            return False
        else:
            os.close(fd)
            os.unlink(file)
            return True
    raise OSError('Unexpected condition testing for writable directory')