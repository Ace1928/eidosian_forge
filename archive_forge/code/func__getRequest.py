from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute.instance_templates import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _getRequest(self, messages, request_data):
    if isinstance(request_data.scope_set, lister.RegionSet):
        return messages.ComputeRegionInstanceTemplatesListRequest
    return messages.ComputeInstanceTemplatesListRequest