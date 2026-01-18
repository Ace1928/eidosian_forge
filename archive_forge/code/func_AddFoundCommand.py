from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.help_search import lookup
def AddFoundCommand(self, command, result):
    """Add a command that is a result.

    Args:
      command: dict, a json representation of a command. MUST already be updated
        with the search results.
      result: search_util.CommandSearchResults, the results object that goes
        with this command.
    """
    self._found_commands_and_results.append((command, result))