import getpass
import locale
import operator
import os
import struct
import sys
import time
from io import BytesIO, TextIOWrapper
from unittest import skipIf
from zope.interface import implementer
from twisted.conch import ls
from twisted.conch.interfaces import ISFTPFile
from twisted.conch.test.test_filetransfer import FileTransferTestAvatar, SFTPTestBase
from twisted.cred import portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.internet.utils import getProcessOutputAndValue, getProcessValue
from twisted.python import log
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
class ListingTests(TestCase):
    """
    Tests for L{lsLine}, the function which generates an entry for a file or
    directory in an SFTP I{ls} command's output.
    """
    if getattr(time, 'tzset', None) is None:
        skip = 'Cannot test timestamp formatting code without time.tzset'

    def setUp(self):
        """
        Patch the L{ls} module's time function so the results of L{lsLine} are
        deterministic.
        """
        self.now = 123456789

        def fakeTime():
            return self.now
        self.patch(ls, 'time', fakeTime)
        if 'TZ' in os.environ:
            self.addCleanup(operator.setitem, os.environ, 'TZ', os.environ['TZ'])
            self.addCleanup(time.tzset)
        else:

            def cleanup():
                try:
                    del os.environ['TZ']
                except KeyError:
                    pass
                time.tzset()
            self.addCleanup(cleanup)

    def _lsInTimezone(self, timezone, stat):
        """
        Call L{ls.lsLine} after setting the timezone to C{timezone} and return
        the result.
        """
        os.environ['TZ'] = timezone
        time.tzset()
        return ls.lsLine('foo', stat)

    def test_oldFile(self):
        """
        A file with an mtime six months (approximately) or more in the past has
        a listing including a low-resolution timestamp.
        """
        then = self.now - 60 * 60 * 24 * 31 * 7
        stat = os.stat_result((0, 0, 0, 0, 0, 0, 0, 0, then, 0))
        self.assertEqual(self._lsInTimezone('America/New_York', stat), '!---------    0 0        0               0 Apr 26  1973 foo')
        self.assertEqual(self._lsInTimezone('Pacific/Auckland', stat), '!---------    0 0        0               0 Apr 27  1973 foo')

    def test_oldSingleDigitDayOfMonth(self):
        """
        A file with a high-resolution timestamp which falls on a day of the
        month which can be represented by one decimal digit is formatted with
        one padding 0 to preserve the columns which come after it.
        """
        then = self.now - 60 * 60 * 24 * 31 * 7 + 60 * 60 * 24 * 5
        stat = os.stat_result((0, 0, 0, 0, 0, 0, 0, 0, then, 0))
        self.assertEqual(self._lsInTimezone('America/New_York', stat), '!---------    0 0        0               0 May 01  1973 foo')
        self.assertEqual(self._lsInTimezone('Pacific/Auckland', stat), '!---------    0 0        0               0 May 02  1973 foo')

    def test_newFile(self):
        """
        A file with an mtime fewer than six months (approximately) in the past
        has a listing including a high-resolution timestamp excluding the year.
        """
        then = self.now - 60 * 60 * 24 * 31 * 3
        stat = os.stat_result((0, 0, 0, 0, 0, 0, 0, 0, then, 0))
        self.assertEqual(self._lsInTimezone('America/New_York', stat), '!---------    0 0        0               0 Aug 28 17:33 foo')
        self.assertEqual(self._lsInTimezone('Pacific/Auckland', stat), '!---------    0 0        0               0 Aug 29 09:33 foo')
    currentLocale = locale.getlocale()
    try:
        try:
            locale.setlocale(locale.LC_ALL, 'es_AR.UTF8')
        except locale.Error:
            localeSkip = True
        else:
            localeSkip = False
    finally:
        locale.setlocale(locale.LC_ALL, currentLocale)

    @skipIf(localeSkip, 'The es_AR.UTF8 locale is not installed.')
    def test_localeIndependent(self):
        """
        The month name in the date is locale independent.
        """
        then = self.now - 60 * 60 * 24 * 31 * 3
        stat = os.stat_result((0, 0, 0, 0, 0, 0, 0, 0, then, 0))
        currentLocale = locale.getlocale()
        locale.setlocale(locale.LC_ALL, 'es_AR.UTF8')
        self.addCleanup(locale.setlocale, locale.LC_ALL, currentLocale)
        self.assertEqual(self._lsInTimezone('America/New_York', stat), '!---------    0 0        0               0 Aug 28 17:33 foo')
        self.assertEqual(self._lsInTimezone('Pacific/Auckland', stat), '!---------    0 0        0               0 Aug 29 09:33 foo')

    def test_newSingleDigitDayOfMonth(self):
        """
        A file with a high-resolution timestamp which falls on a day of the
        month which can be represented by one decimal digit is formatted with
        one padding 0 to preserve the columns which come after it.
        """
        then = self.now - 60 * 60 * 24 * 31 * 3 + 60 * 60 * 24 * 4
        stat = os.stat_result((0, 0, 0, 0, 0, 0, 0, 0, then, 0))
        self.assertEqual(self._lsInTimezone('America/New_York', stat), '!---------    0 0        0               0 Sep 01 17:33 foo')
        self.assertEqual(self._lsInTimezone('Pacific/Auckland', stat), '!---------    0 0        0               0 Sep 02 09:33 foo')