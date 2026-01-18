from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.command_lib.interactive import browser
from prompt_toolkit import keys
from prompt_toolkit.key_binding import manager
import six
class _QuitKeyBinding(_KeyBinding):
    """The quit key binding."""

    def __init__(self, key):
        super(_QuitKeyBinding, self).__init__(key=key, label='quit', help_text='Exit.')

    def Handle(self, event):
        del event
        sys.exit(1)