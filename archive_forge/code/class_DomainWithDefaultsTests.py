import email.message
import email.parser
import errno
import glob
import io
import os
import pickle
import shutil
import signal
import sys
import tempfile
import textwrap
import time
from hashlib import md5
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.portal
import twisted.mail.alias
import twisted.mail.mail
import twisted.mail.maildir
import twisted.mail.protocols
import twisted.mail.relay
import twisted.mail.relaymanager
from twisted import cred, mail
from twisted.internet import address, defer, interfaces, protocol, reactor, task
from twisted.internet.defer import Deferred
from twisted.internet.error import (
from twisted.internet.testing import (
from twisted.mail import pop3, smtp
from twisted.mail.relaymanager import _AttemptManager
from twisted.names import dns
from twisted.names.dns import Record_CNAME, Record_MX, RRHeader
from twisted.names.error import DNSNameError
from twisted.python import failure, log
from twisted.python.filepath import FilePath
from twisted.python.runtime import platformType
from twisted.trial.unittest import TestCase
from twisted.names import client, common, server
@skipIf(platformType != 'posix', 'twisted.mail only works on posix')
class DomainWithDefaultsTests(TestCase):

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def testMethods(self):
        d = {x: x + 10 for x in range(10)}
        d = mail.mail.DomainWithDefaultDict(d, 'Default')
        self.assertEqual(len(d), 10)
        self.assertEqual(list(iter(d)), list(range(10)))
        self.assertEqual(list(d.iterkeys()), list(iter(d)))
        items = list(d.iteritems())
        items.sort()
        self.assertEqual(items, [(x, x + 10) for x in range(10)])
        values = list(d.itervalues())
        values.sort()
        self.assertEqual(values, list(range(10, 20)))
        items = d.items()
        items.sort()
        self.assertEqual(items, [(x, x + 10) for x in range(10)])
        values = d.values()
        values.sort()
        self.assertEqual(values, list(range(10, 20)))
        for x in range(10):
            self.assertEqual(d[x], x + 10)
            self.assertEqual(d.get(x), x + 10)
            self.assertTrue(x in d)
        del d[2], d[4], d[6]
        self.assertEqual(len(d), 7)
        self.assertEqual(d[2], 'Default')
        self.assertEqual(d[4], 'Default')
        self.assertEqual(d[6], 'Default')
        d.update({'a': None, 'b': (), 'c': '*'})
        self.assertEqual(len(d), 10)
        self.assertEqual(d['a'], None)
        self.assertEqual(d['b'], ())
        self.assertEqual(d['c'], '*')
        d.clear()
        self.assertEqual(len(d), 0)
        self.assertEqual(d.setdefault('key', 'value'), 'value')
        self.assertEqual(d['key'], 'value')
        self.assertEqual(d.popitem(), ('key', 'value'))
        self.assertEqual(len(d), 0)
        dcopy = d.copy()
        self.assertEqual(d.domains, dcopy.domains)
        self.assertEqual(d.default, dcopy.default)

    def _stringificationTest(self, stringifier):
        """
        Assert that the class name of a L{mail.mail.DomainWithDefaultDict}
        instance and the string-formatted underlying domain dictionary both
        appear in the string produced by the given string-returning function.

        @type stringifier: one-argument callable
        @param stringifier: either C{str} or C{repr}, to be used to get a
            string to make assertions against.
        """
        domain = mail.mail.DomainWithDefaultDict({}, 'Default')
        self.assertIn(domain.__class__.__name__, stringifier(domain))
        domain['key'] = 'value'
        self.assertIn(str({'key': 'value'}), stringifier(domain))

    def test_str(self):
        """
        L{DomainWithDefaultDict.__str__} should return a string including
        the class name and the domain mapping held by the instance.
        """
        self._stringificationTest(str)

    def test_repr(self):
        """
        L{DomainWithDefaultDict.__repr__} should return a string including
        the class name and the domain mapping held by the instance.
        """
        self._stringificationTest(repr)

    def test_has_keyDeprecation(self):
        """
        has_key is now deprecated.
        """
        sut = mail.mail.DomainWithDefaultDict({}, 'Default')
        sut.has_key('anything')
        message = 'twisted.mail.mail.DomainWithDefaultDict.has_key was deprecated in Twisted 16.3.0. Use the `in` keyword instead.'
        warnings = self.flushWarnings([self.test_has_keyDeprecation])
        self.assertEqual(1, len(warnings))
        self.assertEqual(DeprecationWarning, warnings[0]['category'])
        self.assertEqual(message, warnings[0]['message'])