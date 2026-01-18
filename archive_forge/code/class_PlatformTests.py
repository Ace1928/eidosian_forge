import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
class PlatformTests(SynchronousTestCase):
    """
    Tests for the default L{Platform} initializer.
    """
    isWinNTDeprecationMessage = 'twisted.python.runtime.Platform.isWinNT was deprecated in Twisted 13.0. Use Platform.isWindows instead.'

    def test_isKnown(self) -> None:
        """
        L{Platform.isKnown} returns a boolean indicating whether this is one of
        the L{runtime.knownPlatforms}.
        """
        platform = Platform()
        self.assertTrue(platform.isKnown())

    def test_isVistaConsistency(self) -> None:
        """
        Verify consistency of L{Platform.isVista}: it can only be C{True} if
        L{Platform.isWinNT} and L{Platform.isWindows} are C{True}.
        """
        platform = Platform()
        if platform.isVista():
            self.assertTrue(platform.isWinNT())
            self.assertTrue(platform.isWindows())
            self.assertFalse(platform.isMacOSX())

    def test_isMacOSXConsistency(self) -> None:
        """
        L{Platform.isMacOSX} can only return C{True} if L{Platform.getType}
        returns C{'posix'}.
        """
        platform = Platform()
        if platform.isMacOSX():
            self.assertEqual(platform.getType(), 'posix')

    def test_isLinuxConsistency(self) -> None:
        """
        L{Platform.isLinux} can only return C{True} if L{Platform.getType}
        returns C{'posix'} and L{sys.platform} starts with C{"linux"}.
        """
        platform = Platform()
        if platform.isLinux():
            self.assertTrue(sys.platform.startswith('linux'))

    def test_isWinNT(self) -> None:
        """
        L{Platform.isWinNT} can return only C{False} or C{True} and can not
        return C{True} if L{Platform.getType} is not C{"win32"}.
        """
        platform = Platform()
        isWinNT = platform.isWinNT()
        self.assertIn(isWinNT, (False, True))
        if platform.getType() != 'win32':
            self.assertFalse(isWinNT)
    test_isWinNT.suppress = [SUPRESS(category=DeprecationWarning, message=isWinNTDeprecationMessage)]

    def test_isWinNTDeprecated(self) -> None:
        """
        L{Platform.isWinNT} is deprecated in favor of L{platform.isWindows}.
        """
        platform = Platform()
        platform.isWinNT()
        warnings = self.flushWarnings([self.test_isWinNTDeprecated])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['message'], self.isWinNTDeprecationMessage)

    def test_supportsThreads(self) -> None:
        """
        L{Platform.supportsThreads} returns C{True} if threads can be created in
        this runtime, C{False} otherwise.
        """
        try:
            namedModule('threading')
        except ImportError:
            self.assertFalse(Platform().supportsThreads())
        else:
            self.assertTrue(Platform().supportsThreads())