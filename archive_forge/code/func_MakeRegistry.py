from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.command_lib.interactive import browser
from prompt_toolkit import keys
from prompt_toolkit.key_binding import manager
import six
def MakeRegistry(self):
    """Makes and returns a key binding registry populated with the bindings."""
    m = manager.KeyBindingManager(enable_abort_and_exit_bindings=True, enable_system_bindings=True, enable_search=True, enable_auto_suggest_bindings=True)
    for binding in self.bindings:
        m.registry.add_binding(binding.key, eager=True)(binding.Handle)
    return m.registry