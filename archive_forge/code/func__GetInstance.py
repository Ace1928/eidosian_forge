from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _GetInstance(self, holder, instance_ref):
    request = holder.client.messages.ComputeInstancesGetRequest(**instance_ref.AsDict())
    return holder.client.MakeRequests([(holder.client.apitools_client.instances, 'Get', request)])[0]