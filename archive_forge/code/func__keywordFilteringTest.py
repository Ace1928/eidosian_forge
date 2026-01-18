from __future__ import annotations
import base64
import codecs
import functools
import locale
import os
import uuid
from collections import OrderedDict
from io import BytesIO
from itertools import chain
from typing import Optional, Type
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, error, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.mail import imap4
from twisted.mail.imap4 import MessageSet
from twisted.mail.interfaces import (
from twisted.protocols import loopback
from twisted.python import failure, log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.trial.unittest import SynchronousTestCase, TestCase
def _keywordFilteringTest(self, keyword):
    """
        Helper to implement tests for value filtering of KEYWORD and UNKEYWORD
        queries.

        @param keyword: A native string giving the name of the L{imap4.Query}
            keyword argument to test.
        """
    self.assertEqual(f'({keyword.upper()} twistedrocks)', imap4.Query(**{keyword: 'twisted (){%*"\\] rocks'}))
    self.assertEqual(f'({keyword.upper()} twistedrocks)', imap4.Query(**{keyword: 'twisted %s rocks' % (''.join((chr(ch) for ch in range(33))),)}))