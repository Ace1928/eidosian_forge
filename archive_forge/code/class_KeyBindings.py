from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.command_lib.interactive import browser
from prompt_toolkit import keys
from prompt_toolkit.key_binding import manager
import six
class KeyBindings(object):
    """All key bindings.

  Attributes:
    bindings: The list of key bindings in left to right order.
    help_key: The help visibility key binding. True for ON, false for
      OFF.
    context_key: The command prefix context key that sets the context to the
      command substring from the beginning of the input line to the current
      cursor position.
    web_help_key: The browse key binding that pops up the full reference
      doc in a browser.
    quit_key: The key binding that exits the shell.
  """

    def __init__(self, help_mode=True):
        """Associates keys with handlers. Toggle states are reachable from here."""
        self.help_key = _HelpKeyBinding(keys.Keys.F2, toggle=help_mode)
        self.context_key = _ContextKeyBinding(keys.Keys.F7)
        self.web_help_key = _WebHelpKeyBinding(keys.Keys.F8)
        self.quit_key = _QuitKeyBinding(keys.Keys.F9)
        self.interrupt_signal = _InterruptKeyBinding(keys.Keys.ControlC)
        self.stop_signal = _StopKeyBinding(keys.Keys.ControlZ)
        self.bindings = [self.help_key, self.context_key, self.web_help_key, self.quit_key, self.interrupt_signal, self.stop_signal]

    def MakeRegistry(self):
        """Makes and returns a key binding registry populated with the bindings."""
        m = manager.KeyBindingManager(enable_abort_and_exit_bindings=True, enable_system_bindings=True, enable_search=True, enable_auto_suggest_bindings=True)
        for binding in self.bindings:
            m.registry.add_binding(binding.key, eager=True)(binding.Handle)
        return m.registry

    def Initialize(self, cli):
        """Initialize key binding defaults in the cli."""
        for binding in self.bindings:
            binding.SetMode(cli)