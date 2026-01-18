from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
from six.moves import range  # pylint: disable=redefined-builtin
def SplitInstancesInRequest(request, request_field, max_length=INSTANCES_MAX_LENGTH):
    """Split request into parts according to max_length limit.

  Example:
    requests = SplitInstancesInRequest(
          self.messages.
          ComputeInstanceGroupManagersAbandonInstancesRequest(
              instanceGroupManager=igm_ref.Name(),
              instanceGroupManagersAbandonInstancesRequest=(
                  self.messages.InstanceGroupManagersAbandonInstancesRequest(
                      instances=instances,
                  )
              ),
              project=igm_ref.project,
              zone=igm_ref.zone,
          ), 'instanceGroupManagersAbandonInstancesRequest')

  Then:
    return client.MakeRequests(LiftRequestsList(service, method, requests))

  Args:
    request: _messages.Message, request to split
    request_field: str, name of property inside request holding instances field
    max_length: int, max_length of instances property

  Returns:
    List of requests with instances field length limited by max_length.
  """
    result = []
    all_instances = getattr(request, request_field).instances or []
    n = len(all_instances)
    for i in range(0, n, max_length):
        request_part = encoding.CopyProtoMessage(request)
        field = getattr(request_part, request_field)
        field.instances = all_instances[i:i + max_length]
        result.append(request_part)
    return result