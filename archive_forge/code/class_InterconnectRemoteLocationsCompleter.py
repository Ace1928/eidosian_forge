from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
class InterconnectRemoteLocationsCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(InterconnectRemoteLocationsCompleter, self).__init__(collection='compute.interconnectRemoteLocations', list_command='alpha compute interconnects remote-locations list --uri', **kwargs)