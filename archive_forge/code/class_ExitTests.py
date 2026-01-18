from io import StringIO
from typing import Optional, Union
import twisted.trial.unittest
from ...runner import _exit
from .._exit import ExitStatus, exit
class ExitTests(twisted.trial.unittest.TestCase):
    """
    Tests for L{exit}.
    """

    def setUp(self) -> None:
        self.exit = DummyExit()
        self.patch(_exit, 'sysexit', self.exit)

    def test_exitStatusInt(self) -> None:
        """
        L{exit} given an L{int} status code will pass it to L{sys.exit}.
        """
        status = 1234
        exit(status)
        self.assertEqual(self.exit.arg, status)

    def test_exitConstant(self) -> None:
        """
        L{exit} given a L{ValueConstant} status code passes the corresponding
        value to L{sys.exit}.
        """
        status = ExitStatus.EX_CONFIG
        exit(status)
        self.assertEqual(self.exit.arg, status.value)

    def test_exitMessageZero(self) -> None:
        """
        L{exit} given a status code of zero (C{0}) writes the given message to
        standard output.
        """
        out = StringIO()
        self.patch(_exit, 'stdout', out)
        message = 'Hello, world.'
        exit(0, message)
        self.assertEqual(out.getvalue(), message + '\n')

    def test_exitMessageNonZero(self) -> None:
        """
        L{exit} given a non-zero status code writes the given message to
        standard error.
        """
        out = StringIO()
        self.patch(_exit, 'stderr', out)
        message = 'Hello, world.'
        exit(64, message)
        self.assertEqual(out.getvalue(), message + '\n')