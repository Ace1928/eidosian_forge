from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
class RegionTargetHttpProxiesCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(RegionTargetHttpProxiesCompleter, self).__init__(collection='compute.regionTargetHttpProxies', list_command='compute target-http-proxies list --filter=region:* --uri', **kwargs)