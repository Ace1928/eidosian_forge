from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def _GetNetworkConfig(self, options):
    """Gets a wrapper containing the network config for the node pool.

    Args:
      options: Network config options

    Returns:
      A NetworkConfig object that contains the options for how the network
      for the nodepool needs to be configured.
    """
    if options.pod_ipv4_range is None and options.create_pod_ipv4_range is None and (options.enable_private_nodes is None) and (options.network_performance_config is None) and (options.disable_pod_cidr_overprovision is None) and (options.additional_node_network is None) and (options.additional_pod_network is None):
        return None
    network_config = self.messages.NodeNetworkConfig()
    if options.pod_ipv4_range is not None:
        network_config.podRange = options.pod_ipv4_range
    if options.create_pod_ipv4_range is not None:
        for key in options.create_pod_ipv4_range:
            if key not in ['name', 'range']:
                raise util.Error(CREATE_POD_RANGE_INVALID_KEY_ERROR_MSG.format(key=key))
        network_config.createPodRange = True
        network_config.podRange = options.create_pod_ipv4_range.get('name', None)
        network_config.podIpv4CidrBlock = options.create_pod_ipv4_range.get('range', None)
    if options.enable_private_nodes is not None:
        network_config.enablePrivateNodes = options.enable_private_nodes
    if options.disable_pod_cidr_overprovision is not None:
        network_config.podCidrOverprovisionConfig = self.messages.PodCIDROverprovisionConfig(disable=options.disable_pod_cidr_overprovision)
    if options.additional_node_network is not None:
        network_config.additionalNodeNetworkConfigs = []
        for node_network_option in options.additional_node_network:
            node_network_config_msg = self.messages.AdditionalNodeNetworkConfig()
            node_network_config_msg.network = node_network_option['network']
            node_network_config_msg.subnetwork = node_network_option['subnetwork']
            network_config.additionalNodeNetworkConfigs.append(node_network_config_msg)
    if options.additional_pod_network is not None:
        network_config.additionalPodNetworkConfigs = []
        for pod_network_option in options.additional_pod_network:
            pod_network_config_msg = self.messages.AdditionalPodNetworkConfig()
            pod_network_config_msg.subnetwork = pod_network_option.get('subnetwork', None)
            pod_network_config_msg.secondaryPodRange = pod_network_option['pod-ipv4-range']
            pod_network_config_msg.maxPodsPerNode = self.messages.MaxPodsConstraint(maxPodsPerNode=pod_network_option.get('max-pods-per-node', None))
            network_config.additionalPodNetworkConfigs.append(pod_network_config_msg)
    return network_config