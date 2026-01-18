from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import complete
from cliff.tests import base
class TestCompletionAction(base.TestBase):

    def given_complete_command(self):
        cmd_mgr = commandmanager.CommandManager('cliff.tests')
        app = application.App('testing', '1', cmd_mgr, stdout=FakeStdout())
        sot = complete.CompleteCommand(app, mock.Mock())
        cmd_mgr.add_command('complete', complete.CompleteCommand)
        return (sot, app, cmd_mgr)

    def then_actions_equal(self, actions):
        optstr = ' '.join((opt for action in actions for opt in action.option_strings))
        self.assertEqual('-h --help --name --shell', optstr)

    def test_complete_command_get_actions(self):
        sot, app, cmd_mgr = self.given_complete_command()
        app.interactive_mode = False
        actions = sot.get_actions(['complete'])
        self.then_actions_equal(actions)

    def test_complete_command_get_actions_interactive(self):
        sot, app, cmd_mgr = self.given_complete_command()
        app.interactive_mode = True
        actions = sot.get_actions(['complete'])
        self.then_actions_equal(actions)

    def test_complete_command_take_action(self):
        sot, app, cmd_mgr = self.given_complete_command()
        parsed_args = mock.Mock()
        parsed_args.name = 'test_take'
        parsed_args.shell = 'bash'
        content = app.stdout.content
        self.assertEqual(0, sot.take_action(parsed_args))
        self.assertIn('_test_take()\n', content[0])
        self.assertIn('complete -F _test_take test_take\n', content[-1])
        self.assertIn("  cmds='complete help'\n", content)
        self.assertIn("  cmds_complete='-h --help --name --shell'\n", content)
        self.assertIn("  cmds_help='-h --help'\n", content)

    def test_complete_command_remove_dashes(self):
        sot, app, cmd_mgr = self.given_complete_command()
        parsed_args = mock.Mock()
        parsed_args.name = 'test-take'
        parsed_args.shell = 'bash'
        content = app.stdout.content
        self.assertEqual(0, sot.take_action(parsed_args))
        self.assertIn('_test_take()\n', content[0])
        self.assertIn('complete -F _test_take test-take\n', content[-1])