from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.command_lib.interactive import browser
from prompt_toolkit import keys
from prompt_toolkit.key_binding import manager
import six
def GetHelp(self, markdown=False):
    """Returns the key help text."""
    if not self.help_text:
        return None
    key = self.GetName()
    if markdown:
        key = '*{}*'.format(key)
    return self.help_text.format(key=key)