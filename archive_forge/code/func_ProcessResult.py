from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.command_lib.help_search import lookup
from googlecloudsdk.core.document_renderers import render_document
import six
from six.moves import filter
def ProcessResult(command, results):
    """Helper function to create help text resource for listing results.

  Args:
    command: dict, json representation of command.
    results: CommandSearchResults, result of searching for each term.

  Returns:
    A modified copy of the json command with a summary, and with the dict
        of subcommands replaced with just a list of available subcommands.
  """
    new_command = copy.deepcopy(command)
    if lookup.COMMANDS in six.iterkeys(new_command):
        new_command[lookup.COMMANDS] = sorted([c[lookup.NAME] for c in new_command[lookup.COMMANDS].values() if not c[lookup.IS_HIDDEN]])
    new_command[lookup.RESULTS] = results.FoundTermsMap()
    return new_command