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
def SendAllInstancesRequest(api_holder, method_name, request_template, all_instances_holder_field, igm_ref):
    """Prepare *-instances request with --all-instances flag and format output.

  Args:
    api_holder: Compute API holder.
    method_name: Name of the (region) instance groups managers service method to
      call.
    request_template: Partially filled *-instances request (no instances).
    all_instances_holder_field: Name of the field inside request holding
      allInstances field.
    igm_ref: URL to the target IGM.

  Returns:
    Empty list (for consistency with a command version using list of instances).
  """
    client = api_holder.client
    if igm_ref.Collection() == 'compute.instanceGroupManagers':
        service = client.apitools_client.instanceGroupManagers
    elif igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
        service = client.apitools_client.regionInstanceGroupManagers
    else:
        raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))
    getattr(request_template, all_instances_holder_field).allInstances = True
    errors = []
    client.MakeRequests([(service, method_name, request_template)], errors)
    if errors:
        raise utils.RaiseToolException(errors)
    return []