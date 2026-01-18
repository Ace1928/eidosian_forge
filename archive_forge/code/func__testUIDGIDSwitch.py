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