from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
class TargetPoolsCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(TargetPoolsCompleter, self).__init__(collection='compute.targetPools', list_command='compute target-pools list --uri', **kwargs)