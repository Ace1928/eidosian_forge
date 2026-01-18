from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
class TargetSslProxiesCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(TargetSslProxiesCompleter, self).__init__(collection='compute.targetSslProxies', list_command='compute target-ssl-proxies list --uri', **kwargs)