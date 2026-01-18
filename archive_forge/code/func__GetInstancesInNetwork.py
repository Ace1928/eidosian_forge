from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import json
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.resource_manager import tags as rm_tags
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.network_firewall_policies import convert_terraform
from googlecloudsdk.command_lib.compute.network_firewall_policies import secure_tags_utils
from googlecloudsdk.command_lib.compute.networks import flags as network_flags
from googlecloudsdk.command_lib.resource_manager import endpoint_utils as endpoints
from googlecloudsdk.command_lib.resource_manager import operations
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def _GetInstancesInNetwork(project, network_name, compute_client):
    """Gets instances in the network."""

    def _HasInterfaceMatchingNetwork(instance):
        return len([network_interface.network for network_interface in instance.networkInterfaces if network_interface.network.endswith('/%s' % network_name)])
    messages = compute_client.MESSAGES_MODULE
    instance_aggregations = compute_client.instances.AggregatedList(messages.ComputeInstancesAggregatedListRequest(project=project, includeAllScopes=True, maxResults=500))
    instances_list = [item.value.instances for item in instance_aggregations.items.additionalProperties]
    instances = list(itertools.chain(*instances_list))
    return list(filter(_HasInterfaceMatchingNetwork, instances))