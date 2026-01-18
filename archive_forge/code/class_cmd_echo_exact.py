from ...commands import Command, plugin_cmds, register_command
from .. import TestCaseWithMemoryTransport
class cmd_echo_exact(Command):
    """This command just repeats what it is given.

    It decodes the argument, and then writes it to stdout.
    """
    takes_args = ['text']
    encoding_type = 'exact'

    def run(self, text=None):
        self.outf.write(text)