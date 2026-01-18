from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
class InterconnectLocationsCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(InterconnectLocationsCompleter, self).__init__(collection='compute.interconnectLocations', list_command='alpha compute interconnects attachments list --uri', **kwargs)