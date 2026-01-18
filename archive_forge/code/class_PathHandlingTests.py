import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class PathHandlingTests(TestCase):
    """
    Handling paths.
    """

    def test_Normalizer(self):
        """
        Normalize paths.
        """
        for inp, outp in [('a', ['a']), ('/a', ['a']), ('/', []), ('a/b/c', ['a', 'b', 'c']), ('/a/b/c', ['a', 'b', 'c']), ('/a/', ['a']), ('a/', ['a'])]:
            self.assertEqual(ftp.toSegments([], inp), outp)
        for inp, outp in [('b', ['a', 'b']), ('b/', ['a', 'b']), ('/b', ['b']), ('/b/', ['b']), ('b/c', ['a', 'b', 'c']), ('b/c/', ['a', 'b', 'c']), ('/b/c', ['b', 'c']), ('/b/c/', ['b', 'c'])]:
            self.assertEqual(ftp.toSegments(['a'], inp), outp)
        for inp, outp in [('//', []), ('//a', ['a']), ('a//', ['a']), ('a//b', ['a', 'b'])]:
            self.assertEqual(ftp.toSegments([], inp), outp)
        for inp, outp in [('//', []), ('//b', ['b']), ('b//c', ['a', 'b', 'c'])]:
            self.assertEqual(ftp.toSegments(['a'], inp), outp)
        for inp, outp in [('..', []), ('../', []), ('a/..', ['x']), ('/a/..', []), ('/a/b/..', ['a']), ('/a/b/../', ['a']), ('/a/b/../c', ['a', 'c']), ('/a/b/../c/', ['a', 'c']), ('/a/b/../../c', ['c']), ('/a/b/../../c/', ['c']), ('/a/b/../../c/..', []), ('/a/b/../../c/../', [])]:
            self.assertEqual(ftp.toSegments(['x'], inp), outp)
        for inp in ['..', '../', 'a/../..', 'a/../../', '/..', '/../', '/a/../..', '/a/../../', '/a/b/../../..']:
            self.assertRaises(ftp.InvalidPath, ftp.toSegments, [], inp)
        for inp in ['../..', '../../', '../a/../..']:
            self.assertRaises(ftp.InvalidPath, ftp.toSegments, ['x'], inp)