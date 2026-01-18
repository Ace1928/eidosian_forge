from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.command_lib.interactive import browser
from prompt_toolkit import keys
from prompt_toolkit.key_binding import manager
import six
class _WebHelpKeyBinding(_KeyBinding):
    """The web help key binding."""

    def __init__(self, key):
        super(_WebHelpKeyBinding, self).__init__(key=key, label='web-help', help_text='Opens a web browser tab/window to display the complete man page help for the current command. If there is no active web browser (running in *ssh*(1) for example), then command specific help or *man*(1) help is attempted.')

    def Handle(self, event):
        doc = event.cli.current_buffer.document
        browser.OpenReferencePage(event.cli, doc.text, doc.cursor_position)