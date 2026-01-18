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
def MakeRequestsAndGetStatusPerInstanceFromOperation(client, requests, instances_holder_field, warnings_to_collect, errors_to_collect):
    """Make *-instances requests with feedback per instance.

  Specialized version of MakeRequestsAndGetStatusPerInstance. Checks operations
  for warnings presence to evaluate statuses per instance. Gracefully validated
  requests may produce warnings on operations, indicating instances skipped.
  It would be merged with MakeRequestsAndGetStatusPerInstance after we see
  there's no issues with this implementation.

  Args:
    client: Compute client.
    requests: [(service, method, request)].
    instances_holder_field: name of field inside request holding list of
      instances.
    warnings_to_collect: A list for capturing warnings. If any completed
      operation will contain skipped instances, function will append warning
      suggesting how to find additional details on the operation, warnings
      unrelated to graceful validation will be collected as is.
    errors_to_collect: A list for capturing errors. If any response contains an
      error, it is added to this list.

  Returns:
    See MakeRequestsAndGetStatusPerInstance.
  """
    request_results = []
    for service, method, request in requests:
        errors = []
        operations = client.MakeRequests([(service, method, request)], errors, log_warnings=False, no_followup=True, always_return_operation=True)
        [operation] = operations or [None]
        request_results.append((request, operation, errors))
        errors_to_collect.extend(errors)
    status_per_instance = []
    for request, operation, errors in request_results:
        if errors:
            for instance in getattr(request, instances_holder_field).instances:
                status_per_instance.append({'selfLink': instance, 'instanceName': path_simplifier.Name(instance), 'status': 'FAIL'})
        else:
            if operation.targetLink:
                log.status.write('Updated [{0}].\n'.format(operation.targetLink))
            skipped_instances = ExtractSkippedInstancesAndCollectOtherWarnings(operation, warnings_to_collect)
            for instance in getattr(request, instances_holder_field).instances:
                instance_path = instance[instance.find('/projects/') + 1:]
                validation_error = None
                if instance_path in skipped_instances:
                    instance_status = 'SKIPPED'
                    validation_error = skipped_instances[instance_path]
                else:
                    instance_status = 'SUCCESS'
                status_per_instance.append({'selfLink': instance, 'instanceName': path_simplifier.Name(instance), 'status': instance_status, 'validationError': validation_error})
    return status_per_instance