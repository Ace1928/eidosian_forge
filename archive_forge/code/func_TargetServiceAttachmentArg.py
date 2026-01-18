from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags as addresses_flags
from googlecloudsdk.command_lib.util import completers
def TargetServiceAttachmentArg():
    """Return a resource argument for parsing a target service attachment."""
    target_service_attachment_arg = compute_flags.ResourceArgument(name='--target-service-attachment', required=False, resource_name='target service attachment', regional_collection='compute.serviceAttachments', short_help='Target service attachment that receives the traffic.', detailed_help='Target service attachment that receives the traffic. The target service attachment must be in the same region as the forwarding rule.', region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)
    return target_service_attachment_arg