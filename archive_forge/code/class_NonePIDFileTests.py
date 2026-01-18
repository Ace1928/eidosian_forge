import errno
from functools import wraps
from os import getpid, name as SYSTEM_NAME
from typing import Any, Callable, Optional
from zope.interface.verify import verifyObject
from typing_extensions import NoReturn
import twisted.trial.unittest
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
from ...runner import _pidfile
from .._pidfile import (
class NonePIDFileTests(twisted.trial.unittest.TestCase):
    """
    Tests for L{NonePIDFile}.
    """

    def test_interface(self) -> None:
        """
        L{NonePIDFile} conforms to L{IPIDFile}.
        """
        pidFile = NonePIDFile()
        verifyObject(IPIDFile, pidFile)

    def test_read(self) -> None:
        """
        L{NonePIDFile.read} raises L{NoPIDFound}.
        """
        pidFile = NonePIDFile()
        e = self.assertRaises(NoPIDFound, pidFile.read)
        self.assertEqual(str(e), 'PID file does not exist')

    def test_write(self) -> None:
        """
        L{NonePIDFile._write} raises L{OSError} with an errno of L{errno.EPERM}.
        """
        pidFile = NonePIDFile()
        error = self.assertRaises(OSError, pidFile._write, 0)
        self.assertEqual(error.errno, errno.EPERM)

    def test_writeRunningPID(self) -> None:
        """
        L{NonePIDFile.writeRunningPID} raises L{OSError} with an errno of
        L{errno.EPERM}.
        """
        pidFile = NonePIDFile()
        error = self.assertRaises(OSError, pidFile.writeRunningPID)
        self.assertEqual(error.errno, errno.EPERM)

    def test_remove(self) -> None:
        """
        L{NonePIDFile.remove} raises L{OSError} with an errno of L{errno.EPERM}.
        """
        pidFile = NonePIDFile()
        error = self.assertRaises(OSError, pidFile.remove)
        self.assertEqual(error.errno, errno.ENOENT)

    def test_isRunning(self) -> None:
        """
        L{NonePIDFile.isRunning} returns L{False}.
        """
        pidFile = NonePIDFile()
        self.assertEqual(pidFile.isRunning(), False)

    def test_contextManager(self) -> None:
        """
        When used as a context manager, a L{NonePIDFile} doesn't raise, despite
        not existing.
        """
        pidFile = NonePIDFile()
        with pidFile:
            pass