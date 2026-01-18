from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_ALPHA
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_BETA
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_GA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_ALPHA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_BETA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_GA
def _CreateNodeConfig(messages, flags):
    """Creates node config from parameters, returns None if config is empty."""
    if not (flags.location or flags.machine_type or flags.network or flags.subnetwork or flags.service_account or flags.oauth_scopes or flags.tags or flags.disk_size_gb or flags.use_ip_aliases or flags.cluster_secondary_range_name or flags.network_attachment or flags.services_secondary_range_name or flags.cluster_ipv4_cidr_block or flags.services_ipv4_cidr_block or flags.enable_ip_masq_agent or flags.composer_internal_ipv4_cidr_block):
        return None
    config = messages.NodeConfig(location=flags.location, machineType=flags.machine_type, network=flags.network, subnetwork=flags.subnetwork, serviceAccount=flags.service_account, diskSizeGb=flags.disk_size_gb)
    if flags.network_attachment:
        config.composerNetworkAttachment = flags.network_attachment
    if flags.composer_internal_ipv4_cidr_block:
        config.composerInternalIpv4CidrBlock = flags.composer_internal_ipv4_cidr_block
    if flags.oauth_scopes:
        config.oauthScopes = sorted([s.strip() for s in flags.oauth_scopes])
    if flags.tags:
        config.tags = sorted([t.strip() for t in flags.tags])
    if flags.use_ip_aliases or flags.cluster_secondary_range_name or flags.services_secondary_range_name or flags.cluster_ipv4_cidr_block or flags.services_ipv4_cidr_block:
        config.ipAllocationPolicy = messages.IPAllocationPolicy(useIpAliases=flags.use_ip_aliases, clusterSecondaryRangeName=flags.cluster_secondary_range_name, servicesSecondaryRangeName=flags.services_secondary_range_name, clusterIpv4CidrBlock=flags.cluster_ipv4_cidr_block, servicesIpv4CidrBlock=flags.services_ipv4_cidr_block)
        if flags.max_pods_per_node:
            config.maxPodsPerNode = flags.max_pods_per_node
    if flags.enable_ip_masq_agent:
        config.enableIpMasqAgent = flags.enable_ip_masq_agent
    return config