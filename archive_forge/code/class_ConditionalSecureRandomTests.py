from __future__ import annotations
from typing import Callable
from typing_extensions import NoReturn, Protocol
from twisted.python import randbytes
from twisted.trial import unittest
class ConditionalSecureRandomTests(SecureRandomTestCaseBase, unittest.SynchronousTestCase):
    """
    Test random sources one by one, then remove it to.
    """

    def setUp(self) -> None:
        """
        Create a L{randbytes.RandomFactory} to use in the tests.
        """
        self.factory = randbytes.RandomFactory()

    def errorFactory(self, nbytes: object) -> NoReturn:
        """
        A factory raising an error when a source is not available.
        """
        raise randbytes.SourceNotAvailable()

    def test_osUrandom(self) -> None:
        """
        L{RandomFactory._osUrandom} should work as a random source whenever
        L{os.urandom} is available.
        """
        self._check(self.factory._osUrandom)

    def test_withoutAnything(self) -> None:
        """
        Remove all secure sources and assert it raises a failure. Then try the
        fallback parameter.
        """
        self.factory._osUrandom = self.errorFactory
        self.assertRaises(randbytes.SecureRandomNotAvailable, self.factory.secureRandom, 18)

        def wrapper() -> bytes:
            return self.factory.secureRandom(18, fallback=True)
        s = self.assertWarns(RuntimeWarning, 'urandom unavailable - proceeding with non-cryptographically secure random source', __file__, wrapper)
        self.assertEqual(len(s), 18)