import os
import re
import struct
from unittest import skipIf
from hamcrest import assert_that, equal_to
from twisted.internet import defer
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import StringTransport
from twisted.protocols import loopback
from twisted.python import components
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
class FileTransferTestAvatar(TestAvatar):

    def __init__(self, homeDir):
        TestAvatar.__init__(self)
        self.homeDir = homeDir

    def getHomeDir(self):
        return FilePath(os.getcwd()).preauthChild(self.homeDir.path)