from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def NetworkAttachmentArgument(required=True, plural=False):
    return compute_flags.ResourceArgument(resource_name='network attachment', completer=NetworkAttachmentsCompleter, plural=plural, required=required, regional_collection='compute.networkAttachments', region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)