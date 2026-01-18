from pecan.tests import PecanTestCase
class TestCommandRunner(PecanTestCase):

    def test_commands(self):
        from pecan.commands import ServeCommand, ShellCommand, CreateCommand, CommandRunner
        runner = CommandRunner()
        assert runner.commands['serve'] == ServeCommand
        assert runner.commands['shell'] == ShellCommand
        assert runner.commands['create'] == CreateCommand

    def test_run(self):
        from pecan.commands import CommandRunner
        runner = CommandRunner()
        self.assertRaises(RuntimeError, runner.run, ['serve', 'missing_file.py'])