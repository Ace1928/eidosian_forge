from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
import zlib
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_projector
def _GetAllGuestInventoryGuestAttributes(self, holder, instances):
    client = holder.client
    messages = client.messages
    project = properties.VALUES.core.project.GetOrFail()
    requests = [self._GetGuestAttributesRequest(messages, instance['name'], project, os.path.basename(instance['zone'])) for instance in instances]
    responses = holder.client.AsyncRequests([(client.apitools_client.instances, 'GetGuestAttributes', request) for request in requests])
    for response in filter(None, responses):
        for item in response.queryValue.items:
            if item.key in self._GUEST_ATTRIBUTES_PACKAGE_FIELD_KEYS:
                item.value = zlib.decompress(base64.b64decode(item.value), zlib.MAX_WBITS | 32)
    return responses