from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
def CreateUpdateRequest(release_track, ref, args):
    """Returns an updated request formatted to the right URI endpoint."""
    messages = Messages(ref.GetCollectionInfo().api_version)
    custom_uri = 'projects/{project_id}/locations/{location}'.format(project_id=ref.projectsId, location=args.location)
    bundles_config = messages.BundlesConfig(configControllerConfig=messages.ConfigControllerConfig(enabled=True))
    if release_track == base.ReleaseTrack.ALPHA and args.IsSpecified('experimental_features'):
        bundles_config.configControllerConfig.experimentalFeatures = args.experimental_features
    krm_api_host = messages.KrmApiHost(bundlesConfig=bundles_config)
    if args.use_private_endpoint:
        krm_api_host.usePrivateEndpoint = args.use_private_endpoint
    multiple_cidr_blocks = []
    if args.IsSpecified('man_block') and args.IsSpecified('man_blocks'):
        raise Exception('man_block and man_blocks can not be set at the same time')
    if args.IsSpecified('man_blocks'):
        for cidr_block in args.man_blocks:
            cur_cidr_block = messages.CidrBlock(cidrBlock=cidr_block)
            multiple_cidr_blocks.append(cur_cidr_block)
    man_blocks = messages.MasterAuthorizedNetworksConfig(cidrBlocks=multiple_cidr_blocks)
    if args.full_management:
        full_mgmt_config = messages.FullManagementConfig(clusterCidrBlock=args.cluster_ipv4_cidr_block, clusterNamedRange=args.cluster_named_range, manBlock=args.man_block, masterIpv4CidrBlock=args.master_ipv4_cidr_block, network=args.network, subnet=args.subnet, servicesCidrBlock=args.services_ipv4_cidr_block, servicesNamedRange=args.services_named_range)
        if args.IsSpecified('man_blocks'):
            full_mgmt_config.masterAuthorizedNetworksConfig = man_blocks
        mgmt_config = messages.ManagementConfig(fullManagementConfig=full_mgmt_config)
        krm_api_host.managementConfig = mgmt_config
    else:
        master_ipv4_cidr_block = '172.16.0.128/28'
        if args.master_ipv4_cidr_block is not None:
            master_ipv4_cidr_block = args.master_ipv4_cidr_block
        std_mgmt_config = messages.StandardManagementConfig(clusterCidrBlock=args.cluster_ipv4_cidr_block, clusterNamedRange=args.cluster_named_range, manBlock=args.man_block, masterIpv4CidrBlock=master_ipv4_cidr_block, network=args.network, subnet=args.subnet, servicesCidrBlock=args.services_ipv4_cidr_block, servicesNamedRange=args.services_named_range)
        if args.IsSpecified('man_blocks'):
            std_mgmt_config.masterAuthorizedNetworksConfig = man_blocks
        mgmt_config = messages.ManagementConfig(standardManagementConfig=std_mgmt_config)
        krm_api_host.managementConfig = mgmt_config
    request = messages.KrmapihostingProjectsLocationsKrmApiHostsCreateRequest(parent=custom_uri, krmApiHostId=ref.krmApiHostsId, krmApiHost=krm_api_host)
    return request