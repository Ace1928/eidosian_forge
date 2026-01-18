from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.interconnects.attachments import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.interconnects.attachments import flags as attachment_flags
from googlecloudsdk.command_lib.util.args import labels_util
def _get_attachment(self, args):
    holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
    attachment_ref = self.INTERCONNECT_ATTACHMENT_ARG.ResolveAsResource(args, holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(holder.client))
    return client.InterconnectAttachment(attachment_ref, compute_client=holder.client)