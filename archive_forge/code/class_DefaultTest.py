import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
class DefaultTest(LessTest, LVTest):

    def setUp(self) -> None:
        with fixtures.EnvironmentVariable('PAGER', 'less "-r" +F'):
            self.cmd = command.DefaultPager()

    def test_cmd(self) -> None:
        self.assertEqual(['less', '-r', '+F'], self.cmd.command())

    def test_default_cmd(self) -> None:
        with fixtures.EnvironmentVariable('PAGER'):
            cmd = command.DefaultPager()
        self.assertEqual(command.PlatformPager().command(), cmd.command())