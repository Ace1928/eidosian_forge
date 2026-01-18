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
def GetNotesHelpSection(self, contents=None):
    """Returns the NOTES section with explicit and generated help."""
    if not contents:
        contents = self.detailed_help.get('NOTES')
    notes = _Notes(contents)
    if self.IsHidden():
        notes.AddLine('This command is an internal implementation detail and may change or disappear without notice.')
    notes.AddLine(self.ReleaseTrack().help_note)
    alternates = self.GetExistingAlternativeReleaseTracks()
    if alternates:
        notes.AddLine('{} also available:'.format(text.Pluralize(len(alternates), 'This variant is', 'These variants are')))
        notes.AddLine('')
        for alternate in alternates:
            notes.AddLine('  $ ' + alternate)
            notes.AddLine('')
    return notes.GetContents()