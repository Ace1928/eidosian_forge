import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
class PlatformTest(unittest.TestCase):

    def test_aix_cmd(self) -> None:
        with PlatformFixture('aix'):
            cmd = command.PlatformPager()
            self.assertEqual(['more'], cmd.command())
        with PlatformFixture('aix7'):
            cmd = command.PlatformPager()
            self.assertEqual(['more'], cmd.command())

    def test_linux_cmd(self) -> None:
        with PlatformFixture('linux'):
            cmd = command.PlatformPager()
            self.assertEqual(['less'], cmd.command())

    def test_win32_cmd(self) -> None:
        with PlatformFixture('win32'):
            cmd = command.PlatformPager()
            self.assertEqual(['more.com'], cmd.command())

    def test_cygwin_cmd(self) -> None:
        with PlatformFixture('cygwin'):
            cmd = command.PlatformPager()
            self.assertEqual(['less'], cmd.command())

    def test_macos_cmd(self) -> None:
        with PlatformFixture('darwin'):
            cmd = command.PlatformPager()
            self.assertEqual(['less'], cmd.command())

    def test_sunos_cmd(self) -> None:
        with PlatformFixture('sunos5'):
            cmd = command.PlatformPager()
            self.assertEqual(['less'], cmd.command())

    def test_freebsd_cmd(self) -> None:
        with PlatformFixture('freebsd8'):
            cmd = command.PlatformPager()
            self.assertEqual(['less'], cmd.command())