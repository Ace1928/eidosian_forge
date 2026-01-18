from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.interactive import bindings
from googlecloudsdk.command_lib.interactive import bindings_vi
from googlecloudsdk.command_lib.interactive import completer
from googlecloudsdk.command_lib.interactive import coshell as interactive_coshell
from googlecloudsdk.command_lib.interactive import debug as interactive_debug
from googlecloudsdk.command_lib.interactive import layout
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.command_lib.interactive import style as interactive_style
from googlecloudsdk.command_lib.meta import generate_cli_trees
from googlecloudsdk.core import config as core_config
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from prompt_toolkit import application as pt_application
from prompt_toolkit import auto_suggest
from prompt_toolkit import buffer as pt_buffer
from prompt_toolkit import document
from prompt_toolkit import enums
from prompt_toolkit import filters
from prompt_toolkit import history as pt_history
from prompt_toolkit import interface
from prompt_toolkit import shortcuts
from prompt_toolkit import token
from prompt_toolkit.layout import processors as pt_layout
class CLI(interface.CommandLineInterface):
    """Extends the prompt CLI object to include our state.

  Attributes:
    command_count: Command line serial number, incremented on ctrl-c and Run.
    completer: The interactive completer object.
    config: The interactive shell config object.
    coshell: The shell coprocess object.
    debug: The debugging object.
    parser: The interactive parser object.
    root: The root of the static CLI tree that contains all commands, flags,
      positionals and help doc snippets.
  """

    def __init__(self, config=None, coshell=None, debug=None, root=None, interactive_parser=None, interactive_completer=None, application=None, eventloop=None, output=None):
        super(CLI, self).__init__(application=application, eventloop=eventloop, output=output)
        self.command_count = 0
        self.completer = interactive_completer
        self.config = config
        self.coshell = coshell
        self.debug = debug
        self.parser = interactive_parser
        self.root = root

    def Run(self, text, alternate_screen=False):
        """Runs the command line in text, optionally in an alternate screen.

    This should use an alternate screen but I haven't found the incantations
    to get that working. Currently alternate_screen=True clears the default
    screen so full screen commands, like editors and man or help, have a clean
    slate. Otherwise they may overwrite previous output and end up with a
    garbled mess. The downside is that on return the default screen is
    clobbered. Not too bad right now because this is only used as a fallback
    when the real web browser is inaccessible (for example when running in ssh).

    Args:
      text: The command line string to run.
      alternate_screen: Send output to an alternate screen and restore the
        original screen when done.
    """
        if alternate_screen:
            self.renderer.erase()
        self.coshell.Run(text)
        if alternate_screen:
            self.renderer.erase(leave_alternate_screen=False, erase_title=False)
            self.renderer.request_absolute_cursor_position()
            self._redraw()

    def add_buffer(self, name, buf, focus=False):
        """MONKEYPATCH! Calls the async completer on delete before cursor."""
        super(CLI, self).add_buffer(name, buf, focus)

        def DeleteBeforeCursor(count=1):
            deleted = buf.patch_real_delete_before_cursor(count=count)
            buf.patch_completer_function()
            return deleted
        if buf.complete_while_typing() and buf.delete_before_cursor != DeleteBeforeCursor:
            buf.patch_completer_function = self._async_completers[name]
            buf.patch_real_delete_before_cursor = buf.delete_before_cursor
            buf.delete_before_cursor = DeleteBeforeCursor