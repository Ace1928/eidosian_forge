import os
from binascii import Error as BinasciiError, a2b_base64, b2a_base64
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.internet.defer import Deferred
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial.unittest import TestCase
class PlainTextWithCommentTests(PlainEntryTests):
    """
    Test cases for L{PlainEntry} when parsed from a line with a comment.
    """
    plaintextLine = samplePlaintextLine[:-1] + b' plain text comment.\n'
    hostIPLine = sampleHostIPLine[:-1] + b' text following host/IP line\n'