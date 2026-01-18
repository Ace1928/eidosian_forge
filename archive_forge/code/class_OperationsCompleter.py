from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
class OperationsCompleter(completers.MultiResourceCompleter):

    def __init__(self, **kwargs):
        super(OperationsCompleter, self).__init__(completers=[GlobalOperationsCompleter, RegionalOperationsCompleter, ZonalOperationsCompleter], **kwargs)