from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def _symlinkErrorTest(self, errno: int) -> None:

    def fakeSymlink(source: str, dest: str) -> NoReturn:
        raise OSError(errno, None)
    self.patch(lockfile, 'symlink', fakeSymlink)
    lockf = self.mktemp()
    lock = lockfile.FilesystemLock(lockf)
    exc = self.assertRaises(OSError, lock.lock)
    self.assertEqual(exc.errno, errno)