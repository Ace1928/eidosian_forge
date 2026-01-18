from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import re
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def _SetSectionHelp(self, name, lines):
    """Sets section name help composed of lines.

    Args:
      name: The section name.
      lines: The list of lines in the section.
    """
    while lines and (not lines[0]):
        lines = lines[1:]
    while lines and (not lines[-1]):
        lines = lines[:-1]
    if lines:
        self._sections[name] = '\n'.join(lines)