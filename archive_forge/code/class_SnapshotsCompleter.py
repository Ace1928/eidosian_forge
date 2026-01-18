from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import properties
class SnapshotsCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(SnapshotsCompleter, self).__init__(collection='compute.snapshots', list_command='compute snapshots list --uri', **kwargs)