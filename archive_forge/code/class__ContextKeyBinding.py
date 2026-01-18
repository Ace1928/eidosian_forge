from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.command_lib.interactive import browser
from prompt_toolkit import keys
from prompt_toolkit.key_binding import manager
import six
class _ContextKeyBinding(_KeyBinding):
    """set context key binding."""

    def __init__(self, key):
        super(_ContextKeyBinding, self).__init__(key=key, label='context', help_text="Sets the context for command input, so you won't have to re-type common command prefixes at every prompt. The context is the command line from just after the prompt up to the cursor.\n+\nFor example, if you are about to work with `gcloud compute` for a while, type *gcloud compute* and hit {key}. This will display *gcloud compute* at subsequent prompts until the context is changed.\n+\nHit ctrl-c and {key} to clear the context, or edit a command line and/or move the cursor and hit {key} to set a different context.")

    def Handle(self, event):
        event.cli.config.context = event.cli.current_buffer.document.text_before_cursor