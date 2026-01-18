from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def ApplyIpAddressSelectionPolicyArgs(client, args, backend_service):
    """Applies the IP address selection policy argument to the backend service.

  Args:
    client: The client used by gcloud.
    args: The arguments passed to the gcloud command.
    backend_service: The backend service object.
  """
    if HasIpAddressSelectionPolicyArgs(args):
        backend_service.ipAddressSelectionPolicy = client.messages.BackendService.IpAddressSelectionPolicyValueValuesEnum(args.ip_address_selection_policy)