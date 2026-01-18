from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
class RegionTargetHttpsProxiesCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(RegionTargetHttpsProxiesCompleter, self).__init__(collection='compute.regionTargetHttpsProxies', list_command='compute target-https-proxies list --filter=region:* --uri', **kwargs)