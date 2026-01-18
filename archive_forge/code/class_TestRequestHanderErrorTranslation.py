import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
class TestRequestHanderErrorTranslation(TestCase):
    """Tests for breezy.bzr.smart.request._translate_error."""

    def assertTranslationEqual(self, expected_tuple, error):
        self.assertEqual(expected_tuple, request._translate_error(error))

    def test_NoSuchFile(self):
        self.assertTranslationEqual((b'NoSuchFile', b'path'), transport.NoSuchFile('path'))

    def test_LockContention(self):
        self.assertTranslationEqual((b'LockContention',), errors.LockContention('lock', 'msg'))

    def test_TokenMismatch(self):
        self.assertTranslationEqual((b'TokenMismatch', b'some-token', b'actual-token'), errors.TokenMismatch(b'some-token', b'actual-token'))

    def test_MemoryError(self):
        self.assertTranslationEqual((b'MemoryError',), MemoryError())

    def test_GhostRevisionsHaveNoRevno(self):
        self.assertTranslationEqual((b'GhostRevisionsHaveNoRevno', b'revid1', b'revid2'), errors.GhostRevisionsHaveNoRevno(b'revid1', b'revid2'))

    def test_generic_Exception(self):
        self.assertTranslationEqual((b'error', b'Exception', b''), Exception())

    def test_generic_BzrError(self):
        self.assertTranslationEqual((b'error', b'BzrError', b'some text'), errors.BzrError(msg='some text'))

    def test_generic_zlib_error(self):
        from zlib import error
        msg = 'Error -3 while decompressing data: incorrect data check'
        self.assertTranslationEqual((b'error', b'zlib.error', msg.encode('utf-8')), error(msg))