import errno
import os.path
import shutil
import sys
import warnings
from typing import Iterable, Mapping, MutableMapping, Sequence
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.error import ProcessDone
from twisted.internet.interfaces import IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.python import util
from twisted.python.filepath import FilePath
from twisted.test.test_process import MockOS
from twisted.trial.unittest import FailTest, TestCase
from twisted.trial.util import suppress as SUPPRESS
@skipIf(not getattr(os, 'geteuid', None), 'geteuid/seteuid not available')
class RunAsEffectiveUserTests(TestCase):
    """
    Test for the L{util.runAsEffectiveUser} function.
    """

    def setUp(self):
        self.mockos = MockOS()
        self.patch(os, 'geteuid', self.mockos.geteuid)
        self.patch(os, 'getegid', self.mockos.getegid)
        self.patch(os, 'seteuid', self.mockos.seteuid)
        self.patch(os, 'setegid', self.mockos.setegid)

    def _securedFunction(self, startUID, startGID, wantUID, wantGID):
        """
        Check if wanted UID/GID matched start or saved ones.
        """
        self.assertTrue(wantUID == startUID or wantUID == self.mockos.seteuidCalls[-1])
        self.assertTrue(wantGID == startGID or wantGID == self.mockos.setegidCalls[-1])

    def test_forwardResult(self):
        """
        L{util.runAsEffectiveUser} forwards the result obtained by calling the
        given function
        """
        result = util.runAsEffectiveUser(0, 0, lambda: 1)
        self.assertEqual(result, 1)

    def test_takeParameters(self):
        """
        L{util.runAsEffectiveUser} pass the given parameters to the given
        function.
        """
        result = util.runAsEffectiveUser(0, 0, lambda x: 2 * x, 3)
        self.assertEqual(result, 6)

    def test_takesKeyworkArguments(self):
        """
        L{util.runAsEffectiveUser} pass the keyword parameters to the given
        function.
        """
        result = util.runAsEffectiveUser(0, 0, lambda x, y=1, z=1: x * y * z, 2, z=3)
        self.assertEqual(result, 6)

    def _testUIDGIDSwitch(self, startUID, startGID, wantUID, wantGID, expectedUIDSwitches, expectedGIDSwitches):
        """
        Helper method checking the calls to C{os.seteuid} and C{os.setegid}
        made by L{util.runAsEffectiveUser}, when switching from startUID to
        wantUID and from startGID to wantGID.
        """
        self.mockos.euid = startUID
        self.mockos.egid = startGID
        util.runAsEffectiveUser(wantUID, wantGID, self._securedFunction, startUID, startGID, wantUID, wantGID)
        self.assertEqual(self.mockos.seteuidCalls, expectedUIDSwitches)
        self.assertEqual(self.mockos.setegidCalls, expectedGIDSwitches)
        self.mockos.seteuidCalls = []
        self.mockos.setegidCalls = []

    def test_root(self):
        """
        Check UID/GID switches when current effective UID is root.
        """
        self._testUIDGIDSwitch(0, 0, 0, 0, [], [])
        self._testUIDGIDSwitch(0, 0, 1, 0, [1, 0], [])
        self._testUIDGIDSwitch(0, 0, 0, 1, [], [1, 0])
        self._testUIDGIDSwitch(0, 0, 1, 1, [1, 0], [1, 0])

    def test_UID(self):
        """
        Check UID/GID switches when current effective UID is non-root.
        """
        self._testUIDGIDSwitch(1, 0, 0, 0, [0, 1], [])
        self._testUIDGIDSwitch(1, 0, 1, 0, [], [])
        self._testUIDGIDSwitch(1, 0, 1, 1, [0, 1, 0, 1], [1, 0])
        self._testUIDGIDSwitch(1, 0, 2, 1, [0, 2, 0, 1], [1, 0])

    def test_GID(self):
        """
        Check UID/GID switches when current effective GID is non-root.
        """
        self._testUIDGIDSwitch(0, 1, 0, 0, [], [0, 1])
        self._testUIDGIDSwitch(0, 1, 0, 1, [], [])
        self._testUIDGIDSwitch(0, 1, 1, 1, [1, 0], [])
        self._testUIDGIDSwitch(0, 1, 1, 2, [1, 0], [2, 1])

    def test_UIDGID(self):
        """
        Check UID/GID switches when current effective UID/GID is non-root.
        """
        self._testUIDGIDSwitch(1, 1, 0, 0, [0, 1], [0, 1])
        self._testUIDGIDSwitch(1, 1, 0, 1, [0, 1], [])
        self._testUIDGIDSwitch(1, 1, 1, 0, [0, 1, 0, 1], [0, 1])
        self._testUIDGIDSwitch(1, 1, 1, 1, [], [])
        self._testUIDGIDSwitch(1, 1, 2, 1, [0, 2, 0, 1], [])
        self._testUIDGIDSwitch(1, 1, 1, 2, [0, 1, 0, 1], [2, 1])
        self._testUIDGIDSwitch(1, 1, 2, 2, [0, 2, 0, 1], [2, 1])