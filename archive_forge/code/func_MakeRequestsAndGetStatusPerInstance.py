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
def MakeRequestsAndGetStatusPerInstance(client, requests, instances_holder_field, errors_to_collect):
    """Make *-instances requests with feedback per instance.

  Args:
    client: Compute client.
    requests: [(service, method, request)].
    instances_holder_field: name of field inside request holding list of
      instances.
    errors_to_collect: A list for capturing errors. If any response contains an
      error, it is added to this list.

  Returns:
    A list of request statuses per instance. Requests status is a dictionary
    object, see SendInstancesRequestsAndPostProcessOutputs for details.
  """
    request_results = []
    for service, method, request in requests:
        errors = []
        client.MakeRequests([(service, method, request)], errors)
        request_results.append((request, errors))
        errors_to_collect.extend(errors)
    status_per_instance = []
    for request, errors in request_results:
        if errors:
            instance_status = 'FAIL'
        else:
            instance_status = 'SUCCESS'
        for instance in getattr(request, instances_holder_field).instances:
            status_per_instance.append({'selfLink': instance, 'instanceName': path_simplifier.Name(instance), 'status': instance_status})
    return status_per_instance