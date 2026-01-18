from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
class ZonalOperationsCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(ZonalOperationsCompleter, self).__init__(collection='compute.zoneOperations', list_command='compute operations list --uri --filter=zone:*', **kwargs)