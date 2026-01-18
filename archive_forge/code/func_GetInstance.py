from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
import traceback
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def GetInstance(client, instance_ref):
    request = client.messages.ComputeInstancesGetRequest(**instance_ref.AsDict())
    return client.MakeRequests([(client.apitools_client.instances, 'Get', request)])[0]