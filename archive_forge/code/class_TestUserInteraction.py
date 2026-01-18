from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
class TestUserInteraction(script.TestCaseWithMemoryTransportAndScript):

    def test_confirm_action(self):
        """You can write tests that demonstrate user confirmation.

        Specifically, ScriptRunner does't care if the output line for the
        prompt isn't terminated by a newline from the program; it's implicitly
        terminated by the input.
        """
        commands.builtin_command_registry.register(cmd_test_confirm)
        self.addCleanup(commands.builtin_command_registry.remove, 'test-confirm')
        self.run_script('\n            $ brz test-confirm\n            2>Really do it? ([y]es, [n]o): yes\n            <y\n            Do it!\n            $ brz test-confirm\n            2>Really do it? ([y]es, [n]o): no\n            <n\n            ok, no\n            ')