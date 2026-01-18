from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute.networks.subnets import flags
from googlecloudsdk.core.console import console_io
import ipaddress
import six
def _GetSubnetwork(self, client, subnetwork_ref):
    get_request = (client.apitools_client.subnetworks, 'Get', client.messages.ComputeSubnetworksGetRequest(project=subnetwork_ref.project, region=subnetwork_ref.region, subnetwork=subnetwork_ref.Name()))
    objects = client.MakeRequests([get_request])
    return objects[0] if objects else None