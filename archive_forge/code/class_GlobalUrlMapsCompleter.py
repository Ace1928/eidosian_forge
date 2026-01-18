from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
class GlobalUrlMapsCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(GlobalUrlMapsCompleter, self).__init__(collection='compute.urlMaps', list_command='compute url-maps list --global --uri', **kwargs)