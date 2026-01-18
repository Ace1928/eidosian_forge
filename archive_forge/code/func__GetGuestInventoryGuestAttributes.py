from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
import six
def _GetGuestInventoryGuestAttributes(self, instance_ref):
    try:
        holder = base_classes.ComputeApiHolder(base.ReleaseTrack.GA)
        client = holder.client
        messages = client.messages
        request = messages.ComputeInstancesGetGuestAttributesRequest(instance=instance_ref.Name(), project=instance_ref.project, queryPath='guestInventory/', zone=instance_ref.zone)
        response = client.apitools_client.instances.GetGuestAttributes(request)
        return response.queryValue.items
    except Exception as e:
        if "The resource 'guestInventory/' of type 'Guest Attribute' was not found." in six.text_type(e):
            return []
        raise e