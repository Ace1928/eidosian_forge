from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.command_lib.interactive import browser
from prompt_toolkit import keys
from prompt_toolkit.key_binding import manager
import six
def SetMode(self, cli):
    """Sets the toggle mode in the cli."""
    del cli