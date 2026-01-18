from pecan.tests import PecanTestCase
class TestCommandManager(PecanTestCase):

    def test_commands(self):
        from pecan.commands import ServeCommand, ShellCommand, CreateCommand
        from pecan.commands.base import CommandManager
        m = CommandManager()
        assert m.commands['serve'] == ServeCommand
        assert m.commands['shell'] == ShellCommand
        assert m.commands['create'] == CreateCommand