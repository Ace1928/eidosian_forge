from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import flags as forwarding_rule_flags
class ServiceAttachmentsCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(ServiceAttachmentsCompleter, self).__init__(collection='compute.serviceAttachments', list_command='compute service-attachments list --uri', **kwargs)