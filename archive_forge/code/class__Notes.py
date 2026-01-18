from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import re
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import display
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import text
import six
class _Notes(object):
    """Auto-generated NOTES section helper."""

    def __init__(self, explicit_notes=None):
        self._notes = []
        if explicit_notes:
            self._notes.append(explicit_notes.rstrip())
            self._paragraph = True
        else:
            self._paragraph = False

    def AddLine(self, line):
        """Adds a note line with preceding separator if not empty."""
        if not line:
            if line is None:
                return
        elif self._paragraph:
            self._paragraph = False
            self._notes.append('')
        self._notes.append(line.rstrip())

    def GetContents(self):
        """Returns the notes contents as a single string."""
        return '\n'.join(self._notes) if self._notes else None