from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.help_search import lookup
def RateAll(self):
    """Adds rating to every command found."""
    for command, results in self._found_commands_and_results:
        rating = CommandRater(results, command).Rate()
        command[lookup.RELEVANCE] = rating