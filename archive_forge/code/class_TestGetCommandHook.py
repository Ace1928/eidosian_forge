import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
class TestGetCommandHook(tests.TestCase):

    def test_fires_on_get_cmd_object(self):
        commands.install_bzr_command_hooks()
        hook_calls = []

        class ACommand(commands.Command):
            __doc__ = 'A sample command.'

        def get_cmd(cmd_or_None, cmd_name):
            hook_calls.append(('called', cmd_or_None, cmd_name))
            if cmd_name in ('foo', 'info'):
                return ACommand()
        commands.Command.hooks.install_named_hook('get_command', get_cmd, None)
        cmd = ACommand()
        self.assertEqual([], hook_calls)
        cmd = commands.get_cmd_object('foo')
        self.assertEqual([('called', None, 'foo')], hook_calls)
        self.assertIsInstance(cmd, ACommand)
        del hook_calls[:]
        cmd = commands.get_cmd_object('info')
        self.assertIsInstance(cmd, ACommand)
        self.assertEqual(1, len(hook_calls))
        self.assertEqual('info', hook_calls[0][2])
        self.assertIsInstance(hook_calls[0][1], builtins.cmd_info)