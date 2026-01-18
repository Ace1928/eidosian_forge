from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.command_lib.container import container_command_util as cmd_util
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
def ParseCreateNodePoolOptionsBase(args):
    """Parses the flags provided with the node pool creation command."""
    enable_autorepair = cmd_util.GetAutoRepair(args)
    flags.WarnForNodeModification(args, enable_autorepair)
    flags.ValidateSurgeUpgradeSettings(args)
    metadata = metadata_utils.ConstructMetadataDict(args.metadata, args.metadata_from_file)
    ephemeral_storage_local_ssd = None
    if args.IsKnownAndSpecified('ephemeral_storage_local_ssd'):
        ephemeral_storage_local_ssd = [] if args.ephemeral_storage_local_ssd is None else args.ephemeral_storage_local_ssd
    local_nvme_ssd_block = None
    if args.IsKnownAndSpecified('local_nvme_ssd_block'):
        local_nvme_ssd_block = [] if args.local_nvme_ssd_block is None else args.local_nvme_ssd_block
    return api_adapter.CreateNodePoolOptions(accelerators=args.accelerator, boot_disk_kms_key=args.boot_disk_kms_key, machine_type=args.machine_type, disk_size_gb=utils.BytesToGb(args.disk_size), scopes=args.scopes, node_version=args.node_version, num_nodes=args.num_nodes, local_ssd_count=args.local_ssd_count, local_nvme_ssd_block=local_nvme_ssd_block, ephemeral_storage_local_ssd=ephemeral_storage_local_ssd, tags=args.tags, threads_per_core=args.threads_per_core, labels=args.labels, node_labels=args.node_labels, node_taints=args.node_taints, enable_autoscaling=args.enable_autoscaling, max_nodes=args.max_nodes, min_cpu_platform=args.min_cpu_platform, min_nodes=args.min_nodes, total_max_nodes=args.total_max_nodes, total_min_nodes=args.total_min_nodes, location_policy=args.location_policy, image_type=args.image_type, image=args.image, image_project=args.image_project, image_family=args.image_family, preemptible=args.preemptible, enable_autorepair=enable_autorepair, enable_autoupgrade=cmd_util.GetAutoUpgrade(args), service_account=args.service_account, disk_type=args.disk_type, metadata=metadata, max_pods_per_node=args.max_pods_per_node, enable_autoprovisioning=args.enable_autoprovisioning, workload_metadata=args.workload_metadata, workload_metadata_from_node=args.workload_metadata_from_node, shielded_secure_boot=args.shielded_secure_boot, shielded_integrity_monitoring=args.shielded_integrity_monitoring, reservation_affinity=args.reservation_affinity, reservation=args.reservation, sandbox=args.sandbox, max_surge_upgrade=args.max_surge_upgrade, max_unavailable_upgrade=args.max_unavailable_upgrade, node_group=args.node_group, system_config_from_file=args.system_config_from_file, pod_ipv4_range=args.pod_ipv4_range, create_pod_ipv4_range=args.create_pod_ipv4_range, gvnic=args.enable_gvnic, enable_image_streaming=args.enable_image_streaming, spot=args.spot, enable_confidential_nodes=args.enable_confidential_nodes, enable_blue_green_upgrade=args.enable_blue_green_upgrade, enable_surge_upgrade=args.enable_surge_upgrade, node_pool_soak_duration=args.node_pool_soak_duration, standard_rollout_policy=args.standard_rollout_policy, enable_private_nodes=args.enable_private_nodes, enable_fast_socket=args.enable_fast_socket, logging_variant=args.logging_variant, windows_os_version=args.windows_os_version, additional_node_network=args.additional_node_network, additional_pod_network=args.additional_pod_network, sole_tenant_node_affinity_file=args.sole_tenant_node_affinity_file, containerd_config_from_file=args.containerd_config_from_file, resource_manager_tags=args.resource_manager_tags, enable_insecure_kubelet_readonly_port=args.enable_insecure_kubelet_readonly_port)