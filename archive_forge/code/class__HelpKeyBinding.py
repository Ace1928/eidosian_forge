from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.command_lib.interactive import browser
from prompt_toolkit import keys
from prompt_toolkit.key_binding import manager
import six
class _HelpKeyBinding(_KeyBinding):
    """The help key binding."""

    def __init__(self, key, toggle=True):
        super(_HelpKeyBinding, self).__init__(key=key, label='help', toggle=toggle, status={False: 'OFF', True: 'ON'}, help_text='Toggles the active help section, *ON* when enabled, *OFF* when disabled.')