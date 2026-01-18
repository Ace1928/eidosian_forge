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
def GetExistingAlternativeReleaseTracks(self, value=None):
    """Gets the names for the command in other release tracks.

    Args:
      value: str, Optional value being parsed after the command.

    Returns:
      [str]: The names for the command in other release tracks.
    """
    existing_alternatives = []
    path = self.GetPath()
    if value:
        path.append(value)
    alternates = self._cli_generator.ReplicateCommandPathForAllOtherTracks(path)
    if alternates:
        top_element = self._TopCLIElement()
        for _, command_path in sorted(six.iteritems(alternates), key=lambda x: x[0].prefix or ''):
            alternative_cmd = top_element.LoadSubElementByPath(command_path[1:])
            if alternative_cmd and (not alternative_cmd.IsHidden()):
                existing_alternatives.append(' '.join(command_path))
    return existing_alternatives