from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util import completers
class LocationsCompleter(completers.ListCommandCompleter):
    """The location completer."""

    def __init__(self, **kwargs):
        super(LocationsCompleter, self).__init__(collection='privateca.projects.locations', list_command='privateca locations list --uri', **kwargs)