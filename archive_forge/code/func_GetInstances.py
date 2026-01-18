from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.ssh_utils import GetExternalIPAddress
from googlecloudsdk.command_lib.compute.ssh_utils import GetInternalIPAddress
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from six.moves import zip
def GetInstances(self, client, refs):
    """Fetches instance objects corresponding to the given references."""
    instance_get_requests = []
    for ref in refs:
        request_protobuf = client.messages.ComputeInstancesGetRequest(instance=ref.Name(), zone=ref.zone, project=ref.project)
        instance_get_requests.append((client.apitools_client.instances, 'Get', request_protobuf))
    instances = client.MakeRequests(instance_get_requests)
    return instances