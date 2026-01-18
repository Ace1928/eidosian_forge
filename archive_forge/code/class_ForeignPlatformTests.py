import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
class ForeignPlatformTests(SynchronousTestCase):
    """
    Tests for L{Platform} based overridden initializer values.
    """

    def test_getType(self) -> None:
        """
        If an operating system name is supplied to L{Platform}'s initializer,
        L{Platform.getType} returns the platform type which corresponds to that
        name.
        """
        self.assertEqual(Platform('nt').getType(), 'win32')
        self.assertEqual(Platform('ce').getType(), 'win32')
        self.assertEqual(Platform('posix').getType(), 'posix')
        self.assertEqual(Platform('java').getType(), 'java')

    def test_isMacOSX(self) -> None:
        """
        If a system platform name is supplied to L{Platform}'s initializer, it
        is used to determine the result of L{Platform.isMacOSX}, which returns
        C{True} for C{"darwin"}, C{False} otherwise.
        """
        self.assertTrue(Platform(None, 'darwin').isMacOSX())
        self.assertFalse(Platform(None, 'linux2').isMacOSX())
        self.assertFalse(Platform(None, 'win32').isMacOSX())

    def test_isLinux(self) -> None:
        """
        If a system platform name is supplied to L{Platform}'s initializer, it
        is used to determine the result of L{Platform.isLinux}, which returns
        C{True} for values beginning with C{"linux"}, C{False} otherwise.
        """
        self.assertFalse(Platform(None, 'darwin').isLinux())
        self.assertTrue(Platform(None, 'linux').isLinux())
        self.assertTrue(Platform(None, 'linux2').isLinux())
        self.assertTrue(Platform(None, 'linux3').isLinux())
        self.assertFalse(Platform(None, 'win32').isLinux())