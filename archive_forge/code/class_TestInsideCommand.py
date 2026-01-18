import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
class TestInsideCommand(tests.TestCaseInTempDir):

    def test_command_see_config_overrides(self):

        def run(cmd):
            c = config.GlobalStack()
            self.assertEqual('12', c.get('xx'))
            self.assertEqual('foo', c.get('yy'))
        self.overrideAttr(builtins.cmd_rocks, 'run', run)
        self.run_bzr(['rocks', '-Oxx=12', '-Oyy=foo'])
        c = config.GlobalStack()
        self.assertEqual(None, c.get('xx'))
        self.assertEqual(None, c.get('yy'))