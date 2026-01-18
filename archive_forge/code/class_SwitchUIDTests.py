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
@skipIf(not getattr(os, 'getuid', None), 'getuid/setuid not available')
class SwitchUIDTests(TestCase):
    """
    Tests for L{util.switchUID}.
    """

    def setUp(self):
        self.mockos = MockOS()
        self.patch(util, 'os', self.mockos)
        self.patch(util, 'initgroups', self.initgroups)
        self.initgroupsCalls = []

    def initgroups(self, uid, gid):
        """
        Save L{util.initgroups} calls in C{self.initgroupsCalls}.
        """
        self.initgroupsCalls.append((uid, gid))

    def test_uid(self):
        """
        L{util.switchUID} calls L{util.initgroups} and then C{os.setuid} with
        the given uid.
        """
        util.switchUID(12000, None)
        self.assertEqual(self.initgroupsCalls, [(12000, None)])
        self.assertEqual(self.mockos.actions, [('setuid', 12000)])

    def test_euid(self):
        """
        L{util.switchUID} calls L{util.initgroups} and then C{os.seteuid} with
        the given uid if the C{euid} parameter is set to C{True}.
        """
        util.switchUID(12000, None, True)
        self.assertEqual(self.initgroupsCalls, [(12000, None)])
        self.assertEqual(self.mockos.seteuidCalls, [12000])

    def test_currentUID(self):
        """
        If the current uid is the same as the uid passed to L{util.switchUID},
        then initgroups does not get called, but a warning is issued.
        """
        uid = self.mockos.getuid()
        util.switchUID(uid, None)
        self.assertEqual(self.initgroupsCalls, [])
        self.assertEqual(self.mockos.actions, [])
        currentWarnings = self.flushWarnings([util.switchUID])
        self.assertEqual(len(currentWarnings), 1)
        self.assertIn('tried to drop privileges and setuid %i' % uid, currentWarnings[0]['message'])
        self.assertIn('but uid is already %i' % uid, currentWarnings[0]['message'])

    def test_currentEUID(self):
        """
        If the current euid is the same as the euid passed to L{util.switchUID},
        then initgroups does not get called, but a warning is issued.
        """
        euid = self.mockos.geteuid()
        util.switchUID(euid, None, True)
        self.assertEqual(self.initgroupsCalls, [])
        self.assertEqual(self.mockos.seteuidCalls, [])
        currentWarnings = self.flushWarnings([util.switchUID])
        self.assertEqual(len(currentWarnings), 1)
        self.assertIn('tried to drop privileges and seteuid %i' % euid, currentWarnings[0]['message'])
        self.assertIn('but euid is already %i' % euid, currentWarnings[0]['message'])