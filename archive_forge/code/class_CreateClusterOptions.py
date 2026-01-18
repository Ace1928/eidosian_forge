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
class CreateClusterOptions(object):
    """Options to pass to CreateCluster."""

    def __init__(self, node_machine_type=None, node_source_image=None, node_disk_size_gb=None, scopes=None, num_nodes=None, additional_zones=None, node_locations=None, user=None, password=None, cluster_version=None, node_version=None, network=None, cluster_ipv4_cidr=None, enable_cloud_logging=None, enable_cloud_monitoring=None, enable_stackdriver_kubernetes=None, enable_logging_monitoring_system_only=None, enable_workload_monitoring_eap=None, subnetwork=None, addons=None, istio_config=None, cloud_run_config=None, local_ssd_count=None, local_ssd_volume_configs=None, local_nvme_ssd_block=None, ephemeral_storage=None, ephemeral_storage_local_ssd=None, boot_disk_kms_key=None, node_pool_name=None, tags=None, autoprovisioning_network_tags=None, node_labels=None, node_taints=None, enable_autoscaling=None, min_nodes=None, max_nodes=None, total_min_nodes=None, total_max_nodes=None, location_policy=None, image_type=None, image=None, image_project=None, image_family=None, issue_client_certificate=None, max_nodes_per_pool=None, enable_kubernetes_alpha=None, alpha_cluster_feature_gates=None, enable_cloud_run_alpha=None, preemptible=None, spot=None, placement_type=None, placement_policy=None, enable_queued_provisioning=None, enable_autorepair=None, enable_autoupgrade=None, service_account=None, enable_master_authorized_networks=None, master_authorized_networks=None, enable_legacy_authorization=None, labels=None, disk_type=None, enable_network_policy=None, enable_l4_ilb_subsetting=None, services_ipv4_cidr=None, enable_ip_alias=None, create_subnetwork=None, cluster_secondary_range_name=None, services_secondary_range_name=None, accelerators=None, enable_binauthz=None, binauthz_evaluation_mode=None, binauthz_policy_bindings=None, min_cpu_platform=None, workload_metadata=None, workload_metadata_from_node=None, maintenance_window=None, enable_pod_security_policy=None, allow_route_overlap=None, private_cluster=None, enable_private_nodes=None, enable_private_endpoint=None, master_ipv4_cidr=None, tpu_ipv4_cidr=None, enable_tpu=None, enable_tpu_service_networking=None, default_max_pods_per_node=None, max_pods_per_node=None, resource_usage_bigquery_dataset=None, security_group=None, enable_private_ipv6_access=None, enable_intra_node_visibility=None, enable_vertical_pod_autoscaling=None, enable_experimental_vertical_pod_autoscaling=None, security_profile=None, security_profile_runtime_rules=None, autoscaling_profile=None, database_encryption_key=None, metadata=None, enable_network_egress_metering=None, enable_resource_consumption_metering=None, workload_pool=None, identity_provider=None, enable_workload_certificates=None, enable_mesh_certificates=None, enable_alts=None, enable_gke_oidc=None, enable_identity_service=None, enable_shielded_nodes=None, linux_sysctls=None, disable_default_snat=None, dataplane_v2=None, enable_dataplane_v2_metrics=None, disable_dataplane_v2_metrics=None, enable_dataplane_v2_flow_observability=None, disable_dataplane_v2_flow_observability=None, dataplane_v2_observability_mode=None, shielded_secure_boot=None, shielded_integrity_monitoring=None, system_config_from_file=None, maintenance_window_start=None, maintenance_window_end=None, maintenance_window_recurrence=None, enable_cost_allocation=None, max_surge_upgrade=None, max_unavailable_upgrade=None, enable_autoprovisioning=None, autoprovisioning_config_file=None, autoprovisioning_service_account=None, autoprovisioning_scopes=None, autoprovisioning_locations=None, min_cpu=None, max_cpu=None, min_memory=None, max_memory=None, min_accelerator=None, max_accelerator=None, autoprovisioning_image_type=None, autoprovisioning_max_surge_upgrade=None, autoprovisioning_max_unavailable_upgrade=None, enable_autoprovisioning_autoupgrade=None, enable_autoprovisioning_autorepair=None, reservation_affinity=None, reservation=None, autoprovisioning_min_cpu_platform=None, enable_master_global_access=None, gvnic=None, enable_master_metrics=None, master_logs=None, release_channel=None, notification_config=None, autopilot=None, private_ipv6_google_access_type=None, enable_confidential_nodes=None, enable_confidential_storage=None, cluster_dns=None, cluster_dns_scope=None, cluster_dns_domain=None, additive_vpc_scope_dns_domain=None, disable_additive_vpc_scope=None, kubernetes_objects_changes_target=None, kubernetes_objects_snapshots_target=None, enable_gcfs=None, enable_image_streaming=None, private_endpoint_subnetwork=None, cross_connect_subnetworks=None, enable_service_externalips=None, threads_per_core=None, logging=None, monitoring=None, enable_managed_prometheus=None, maintenance_interval=None, disable_pod_cidr_overprovision=None, stack_type=None, ipv6_access_type=None, enable_workload_config_audit=None, pod_autoscaling_direct_metrics_opt_in=None, enable_workload_vulnerability_scanning=None, enable_autoprovisioning_surge_upgrade=None, enable_autoprovisioning_blue_green_upgrade=None, autoprovisioning_standard_rollout_policy=None, autoprovisioning_node_pool_soak_duration=None, enable_google_cloud_access=None, managed_config=None, fleet_project=None, enable_fleet=None, gateway_api=None, logging_variant=None, enable_multi_networking=None, enable_security_posture=None, enable_nested_virtualization=None, performance_monitoring_unit=None, network_performance_config=None, enable_insecure_kubelet_readonly_port=None, autoprovisioning_enable_insecure_kubelet_readonly_port=None, enable_k8s_beta_apis=None, security_posture=None, workload_vulnerability_scanning=None, enable_runtime_vulnerability_insight=None, enable_dns_endpoint=None, workload_policies=None, enable_fqdn_network_policy=None, host_maintenance_interval=None, in_transit_encryption=None, containerd_config_from_file=None, resource_manager_tags=None, autoprovisioning_resource_manager_tags=None, enable_secret_manager=None, enable_cilium_clusterwide_network_policy=None, storage_pools=None):
        self.node_machine_type = node_machine_type
        self.node_source_image = node_source_image
        self.node_disk_size_gb = node_disk_size_gb
        self.scopes = scopes
        self.num_nodes = num_nodes
        self.additional_zones = additional_zones
        self.node_locations = node_locations
        self.user = user
        self.password = password
        self.cluster_version = cluster_version
        self.node_version = node_version
        self.network = network
        self.cluster_ipv4_cidr = cluster_ipv4_cidr
        self.enable_cloud_logging = enable_cloud_logging
        self.enable_cloud_monitoring = enable_cloud_monitoring
        self.enable_stackdriver_kubernetes = enable_stackdriver_kubernetes
        self.enable_logging_monitoring_system_only = enable_logging_monitoring_system_only
        self.enable_workload_monitoring_eap = (enable_workload_monitoring_eap,)
        self.subnetwork = subnetwork
        self.addons = addons
        self.istio_config = istio_config
        self.cloud_run_config = cloud_run_config
        self.local_ssd_count = local_ssd_count
        self.local_ssd_volume_configs = local_ssd_volume_configs
        self.ephemeral_storage = ephemeral_storage
        self.ephemeral_storage_local_ssd = ephemeral_storage_local_ssd
        self.local_nvme_ssd_block = local_nvme_ssd_block
        self.boot_disk_kms_key = boot_disk_kms_key
        self.node_pool_name = node_pool_name
        self.tags = tags
        self.autoprovisioning_network_tags = autoprovisioning_network_tags
        self.node_labels = node_labels
        self.node_taints = node_taints
        self.enable_autoscaling = enable_autoscaling
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.total_min_nodes = total_min_nodes
        self.total_max_nodes = total_max_nodes
        self.location_policy = location_policy
        self.image_type = image_type
        self.image = image
        self.image_project = image_project
        self.image_family = image_family
        self.max_nodes_per_pool = max_nodes_per_pool
        self.enable_kubernetes_alpha = enable_kubernetes_alpha
        self.alpha_cluster_feature_gates = alpha_cluster_feature_gates
        self.enable_cloud_run_alpha = enable_cloud_run_alpha
        self.preemptible = preemptible
        self.spot = spot
        self.placement_type = placement_type
        self.placement_policy = placement_policy
        self.enable_queued_provisioning = enable_queued_provisioning
        self.enable_autorepair = enable_autorepair
        self.enable_autoupgrade = enable_autoupgrade
        self.service_account = service_account
        self.enable_master_authorized_networks = enable_master_authorized_networks
        self.master_authorized_networks = master_authorized_networks
        self.enable_legacy_authorization = enable_legacy_authorization
        self.enable_network_policy = enable_network_policy
        self.enable_l4_ilb_subsetting = enable_l4_ilb_subsetting
        self.labels = labels
        self.disk_type = disk_type
        self.services_ipv4_cidr = services_ipv4_cidr
        self.enable_ip_alias = enable_ip_alias
        self.create_subnetwork = create_subnetwork
        self.cluster_secondary_range_name = cluster_secondary_range_name
        self.services_secondary_range_name = services_secondary_range_name
        self.accelerators = accelerators
        self.enable_binauthz = enable_binauthz
        self.binauthz_evaluation_mode = binauthz_evaluation_mode
        self.binauthz_policy_bindings = binauthz_policy_bindings
        self.min_cpu_platform = min_cpu_platform
        self.workload_metadata = workload_metadata
        self.workload_metadata_from_node = workload_metadata_from_node
        self.maintenance_window = maintenance_window
        self.enable_pod_security_policy = enable_pod_security_policy
        self.allow_route_overlap = allow_route_overlap
        self.private_cluster = private_cluster
        self.enable_private_nodes = enable_private_nodes
        self.enable_private_endpoint = enable_private_endpoint
        self.master_ipv4_cidr = master_ipv4_cidr
        self.tpu_ipv4_cidr = tpu_ipv4_cidr
        self.enable_tpu_service_networking = enable_tpu_service_networking
        self.enable_tpu = enable_tpu
        self.issue_client_certificate = issue_client_certificate
        self.default_max_pods_per_node = default_max_pods_per_node
        self.max_pods_per_node = max_pods_per_node
        self.resource_usage_bigquery_dataset = resource_usage_bigquery_dataset
        self.security_group = security_group
        self.enable_private_ipv6_access = enable_private_ipv6_access
        self.enable_intra_node_visibility = enable_intra_node_visibility
        self.enable_vertical_pod_autoscaling = enable_vertical_pod_autoscaling
        self.enable_experimental_vertical_pod_autoscaling = enable_experimental_vertical_pod_autoscaling
        self.security_profile = security_profile
        self.security_profile_runtime_rules = security_profile_runtime_rules
        self.autoscaling_profile = autoscaling_profile
        self.database_encryption_key = database_encryption_key
        self.metadata = metadata
        self.enable_network_egress_metering = enable_network_egress_metering
        self.enable_resource_consumption_metering = enable_resource_consumption_metering
        self.workload_pool = workload_pool
        self.identity_provider = identity_provider
        self.enable_workload_certificates = enable_workload_certificates
        self.enable_mesh_certificates = enable_mesh_certificates
        self.enable_alts = enable_alts
        self.enable_gke_oidc = enable_gke_oidc
        self.enable_identity_service = enable_identity_service
        self.enable_shielded_nodes = enable_shielded_nodes
        self.linux_sysctls = linux_sysctls
        self.disable_default_snat = disable_default_snat
        self.dataplane_v2 = dataplane_v2
        self.enable_dataplane_v2_metrics = enable_dataplane_v2_metrics
        self.disable_dataplane_v2_metrics = disable_dataplane_v2_metrics
        self.enable_dataplane_v2_flow_observability = enable_dataplane_v2_flow_observability
        self.disable_dataplane_v2_flow_observability = disable_dataplane_v2_flow_observability
        self.dataplane_v2_observability_mode = dataplane_v2_observability_mode
        self.shielded_secure_boot = shielded_secure_boot
        self.shielded_integrity_monitoring = shielded_integrity_monitoring
        self.system_config_from_file = system_config_from_file
        self.maintenance_window_start = maintenance_window_start
        self.maintenance_window_end = maintenance_window_end
        self.maintenance_window_recurrence = maintenance_window_recurrence
        self.enable_cost_allocation = enable_cost_allocation
        self.max_surge_upgrade = max_surge_upgrade
        self.max_unavailable_upgrade = max_unavailable_upgrade
        self.enable_autoprovisioning = enable_autoprovisioning
        self.autoprovisioning_config_file = autoprovisioning_config_file
        self.autoprovisioning_service_account = autoprovisioning_service_account
        self.autoprovisioning_scopes = autoprovisioning_scopes
        self.autoprovisioning_locations = autoprovisioning_locations
        self.min_cpu = min_cpu
        self.max_cpu = max_cpu
        self.min_memory = min_memory
        self.max_memory = max_memory
        self.min_accelerator = min_accelerator
        self.max_accelerator = max_accelerator
        self.autoprovisioning_image_type = autoprovisioning_image_type
        self.autoprovisioning_max_surge_upgrade = autoprovisioning_max_surge_upgrade
        self.autoprovisioning_max_unavailable_upgrade = autoprovisioning_max_unavailable_upgrade
        self.enable_autoprovisioning_autoupgrade = enable_autoprovisioning_autoupgrade
        self.enable_autoprovisioning_autorepair = enable_autoprovisioning_autorepair
        self.reservation_affinity = reservation_affinity
        self.reservation = reservation
        self.autoprovisioning_min_cpu_platform = autoprovisioning_min_cpu_platform
        self.enable_master_global_access = enable_master_global_access
        self.gvnic = gvnic
        self.enable_master_metrics = enable_master_metrics
        self.master_logs = master_logs
        self.release_channel = release_channel
        self.notification_config = notification_config
        self.autopilot = autopilot
        self.private_ipv6_google_access_type = private_ipv6_google_access_type
        self.enable_confidential_nodes = enable_confidential_nodes
        self.enable_confidential_storage = enable_confidential_storage
        self.storage_pools = storage_pools
        self.cluster_dns = cluster_dns
        self.cluster_dns_scope = cluster_dns_scope
        self.cluster_dns_domain = cluster_dns_domain
        self.additive_vpc_scope_dns_domain = additive_vpc_scope_dns_domain
        self.disable_additive_vpc_scope = disable_additive_vpc_scope
        self.kubernetes_objects_changes_target = kubernetes_objects_changes_target
        self.kubernetes_objects_snapshots_target = kubernetes_objects_snapshots_target
        self.enable_gcfs = enable_gcfs
        self.enable_image_streaming = enable_image_streaming
        self.private_endpoint_subnetwork = private_endpoint_subnetwork
        self.cross_connect_subnetworks = cross_connect_subnetworks
        self.enable_service_externalips = enable_service_externalips
        self.threads_per_core = threads_per_core
        self.enable_nested_virtualization = enable_nested_virtualization
        self.performance_monitoring_unit = performance_monitoring_unit
        self.logging = logging
        self.monitoring = monitoring
        self.enable_managed_prometheus = enable_managed_prometheus
        self.maintenance_interval = maintenance_interval
        self.disable_pod_cidr_overprovision = disable_pod_cidr_overprovision
        self.stack_type = stack_type
        self.ipv6_access_type = ipv6_access_type
        self.enable_workload_config_audit = enable_workload_config_audit
        self.pod_autoscaling_direct_metrics_opt_in = pod_autoscaling_direct_metrics_opt_in
        self.enable_workload_vulnerability_scanning = enable_workload_vulnerability_scanning
        self.enable_autoprovisioning_surge_upgrade = enable_autoprovisioning_surge_upgrade
        self.enable_autoprovisioning_blue_green_upgrade = enable_autoprovisioning_blue_green_upgrade
        self.autoprovisioning_standard_rollout_policy = autoprovisioning_standard_rollout_policy
        self.autoprovisioning_node_pool_soak_duration = autoprovisioning_node_pool_soak_duration
        self.enable_google_cloud_access = enable_google_cloud_access
        self.managed_config = managed_config
        self.fleet_project = fleet_project
        self.enable_fleet = enable_fleet
        self.gateway_api = gateway_api
        self.logging_variant = logging_variant
        self.enable_multi_networking = enable_multi_networking
        self.enable_security_posture = enable_security_posture
        self.network_performance_config = network_performance_config
        self.enable_insecure_kubelet_readonly_port = enable_insecure_kubelet_readonly_port
        self.autoprovisioning_enable_insecure_kubelet_readonly_port = autoprovisioning_enable_insecure_kubelet_readonly_port
        self.enable_k8s_beta_apis = enable_k8s_beta_apis
        self.security_posture = security_posture
        self.workload_vulnerability_scanning = workload_vulnerability_scanning
        self.enable_runtime_vulnerability_insight = enable_runtime_vulnerability_insight
        self.enable_dns_endpoint = enable_dns_endpoint
        self.workload_policies = workload_policies
        self.enable_fqdn_network_policy = enable_fqdn_network_policy
        self.host_maintenance_interval = host_maintenance_interval
        self.in_transit_encryption = in_transit_encryption
        self.containerd_config_from_file = containerd_config_from_file
        self.resource_manager_tags = resource_manager_tags
        self.autoprovisioning_resource_manager_tags = autoprovisioning_resource_manager_tags
        self.enable_secret_manager = enable_secret_manager
        self.enable_cilium_clusterwide_network_policy = enable_cilium_clusterwide_network_policy