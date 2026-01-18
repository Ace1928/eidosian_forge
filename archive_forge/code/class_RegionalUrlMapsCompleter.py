from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
class RegionalUrlMapsCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(RegionalUrlMapsCompleter, self).__init__(collection='compute.regionUrlMaps', list_command='compute url-maps list --filter=region:* --uri', **kwargs)