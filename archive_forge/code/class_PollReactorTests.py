from __future__ import annotations
import select
import sys
from typing import Callable
from twisted.internet import default
from twisted.internet.default import _getInstallFunction, install
from twisted.internet.interfaces import IReactorCore
from twisted.internet.test.test_main import NoReactor
from twisted.python.reflect import requireModule
from twisted.python.runtime import Platform
from twisted.trial.unittest import SynchronousTestCase
class PollReactorTests(SynchronousTestCase):
    """
    Tests for the cases of L{twisted.internet.default._getInstallFunction}
    in which it picks the poll(2) or epoll(7)-based reactors.
    """

    def assertIsPoll(self, install: Callable[..., object]) -> None:
        """
        Assert the given function will install the poll() reactor, or select()
        if poll() is unavailable.
        """
        if hasattr(select, 'poll'):
            self.assertEqual(install.__module__, 'twisted.internet.pollreactor')
        else:
            self.assertEqual(install.__module__, 'twisted.internet.selectreactor')

    def test_unix(self) -> None:
        """
        L{_getInstallFunction} chooses the poll reactor on arbitrary Unix
        platforms, falling back to select(2) if it is unavailable.
        """
        install = _getInstallFunction(unix)
        self.assertIsPoll(install)

    def test_linux(self) -> None:
        """
        L{_getInstallFunction} chooses the epoll reactor on Linux, or poll if
        epoll is unavailable.
        """
        install = _getInstallFunction(linux)
        if requireModule('twisted.internet.epollreactor') is None:
            self.assertIsPoll(install)
        else:
            self.assertEqual(install.__module__, 'twisted.internet.epollreactor')