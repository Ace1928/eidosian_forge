from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import flags as forwarding_rule_flags
def ServiceAttachmentArgument(required=True, plural=False):
    return compute_flags.ResourceArgument(resource_name='service attachment', completer=ServiceAttachmentsCompleter, plural=plural, required=required, regional_collection='compute.serviceAttachments', region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)