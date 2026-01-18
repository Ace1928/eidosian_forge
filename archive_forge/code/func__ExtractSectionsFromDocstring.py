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
def _ExtractSectionsFromDocstring(self, docstring):
    """Extracts section help from the command docstring."""
    name = 'DESCRIPTION'
    lines = []
    for line in textwrap.dedent(docstring).strip().splitlines():
        if len(line) >= 4 and line.startswith('## '):
            self._SetSectionHelp(name, lines)
            name = line[3:]
            lines = []
        else:
            lines.append(line)
    self._SetSectionHelp(name, lines)