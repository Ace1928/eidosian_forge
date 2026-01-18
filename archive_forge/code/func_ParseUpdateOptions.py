from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import container_command_util
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from six.moves import input  # pylint: disable=redefined-builtin
def ParseUpdateOptions(self, args, locations):
    get_default = lambda key: getattr(args, key)
    flags.ValidateNotificationConfigFlag(args)
    opts = container_command_util.ParseUpdateOptionsBase(args, locations)
    opts.autoscaling_profile = args.autoscaling_profile
    opts.enable_pod_security_policy = args.enable_pod_security_policy
    opts.resource_usage_bigquery_dataset = args.resource_usage_bigquery_dataset
    opts.clear_resource_usage_bigquery_dataset = args.clear_resource_usage_bigquery_dataset
    opts.security_profile = args.security_profile
    opts.istio_config = args.istio_config
    opts.cloud_run_config = flags.GetLegacyCloudRunFlag('{}_config', args, get_default)
    opts.enable_intra_node_visibility = args.enable_intra_node_visibility
    opts.enable_network_egress_metering = args.enable_network_egress_metering
    opts.enable_resource_consumption_metering = args.enable_resource_consumption_metering
    opts.enable_workload_certificates = args.enable_workload_certificates
    opts.enable_alts = args.enable_alts
    opts.enable_experimental_vertical_pod_autoscaling = args.enable_experimental_vertical_pod_autoscaling
    flags.ValidateIstioConfigUpdateArgs(args.istio_config, args.disable_addons)
    flags.ValidateCloudRunConfigUpdateArgs(opts.cloud_run_config, args.disable_addons)
    if args.disable_addons and api_adapter.NODELOCALDNS in args.disable_addons:
        console_io.PromptContinue(message='Enabling/Disabling NodeLocal DNSCache causes a re-creation of all cluster nodes at versions 1.15 or above. This operation is long-running and will block other operations on the cluster (including delete) until it has run to completion.', cancel_on_no=True)
    opts.enable_stackdriver_kubernetes = args.enable_stackdriver_kubernetes
    opts.enable_logging_monitoring_system_only = args.enable_logging_monitoring_system_only
    opts.no_master_logs = args.no_master_logs
    opts.master_logs = args.master_logs
    opts.enable_master_metrics = args.enable_master_metrics
    opts.release_channel = args.release_channel
    opts.enable_tpu = args.enable_tpu
    opts.tpu_ipv4_cidr = args.tpu_ipv4_cidr
    opts.enable_tpu_service_networking = args.enable_tpu_service_networking
    opts.identity_provider = args.identity_provider
    opts.enable_shielded_nodes = args.enable_shielded_nodes
    opts.disable_default_snat = args.disable_default_snat
    opts.enable_cost_allocation = args.enable_cost_allocation
    opts.enable_master_global_access = args.enable_master_global_access
    opts.notification_config = args.notification_config
    opts.kubernetes_objects_changes_target = args.kubernetes_objects_changes_target
    opts.kubernetes_objects_snapshots_target = args.kubernetes_objects_snapshots_target
    opts.enable_gke_oidc = args.enable_gke_oidc
    opts.enable_identity_service = args.enable_identity_service
    opts.enable_workload_monitoring_eap = args.enable_workload_monitoring_eap
    opts.enable_managed_prometheus = args.enable_managed_prometheus
    opts.disable_managed_prometheus = args.disable_managed_prometheus
    opts.disable_autopilot = args.disable_autopilot
    opts.enable_l4_ilb_subsetting = args.enable_l4_ilb_subsetting
    if opts.enable_l4_ilb_subsetting:
        console_io.PromptContinue(message='Enabling L4 ILB Subsetting is a one-way operation.Once enabled, this configuration cannot be disabled.Existing ILB services should be recreated to use Subsetting.', cancel_on_no=True)
    opts.cluster_dns = args.cluster_dns
    opts.cluster_dns_scope = args.cluster_dns_scope
    opts.cluster_dns_domain = args.cluster_dns_domain
    opts.disable_additive_vpc_scope = args.disable_additive_vpc_scope
    opts.additive_vpc_scope_dns_domain = args.additive_vpc_scope_dns_domain
    if opts.cluster_dns and opts.cluster_dns.lower() == 'clouddns':
        console_io.PromptContinue(message='All the node-pools in the cluster need to be re-created by the user to start using Cloud DNS for DNS lookups. It is highly recommended to complete this step shortly after enabling Cloud DNS.', cancel_on_no=True)
    opts.enable_service_externalips = args.enable_service_externalips
    opts.security_group = args.security_group
    opts.enable_gcfs = args.enable_gcfs
    opts.autoprovisioning_network_tags = args.autoprovisioning_network_tags
    opts.enable_image_streaming = args.enable_image_streaming
    opts.maintenance_interval = args.maintenance_interval
    opts.dataplane_v2 = args.enable_dataplane_v2
    opts.enable_dataplane_v2_metrics = args.enable_dataplane_v2_metrics
    opts.disable_dataplane_v2_metrics = args.disable_dataplane_v2_metrics
    opts.enable_dataplane_v2_flow_observability = args.enable_dataplane_v2_flow_observability
    opts.disable_dataplane_v2_flow_observability = args.disable_dataplane_v2_flow_observability
    opts.dataplane_v2_observability_mode = args.dataplane_v2_observability_mode
    opts.enable_workload_config_audit = args.enable_workload_config_audit
    opts.pod_autoscaling_direct_metrics_opt_in = args.pod_autoscaling_direct_metrics_opt_in
    opts.enable_workload_vulnerability_scanning = args.enable_workload_vulnerability_scanning
    opts.enable_private_endpoint = args.enable_private_endpoint
    opts.enable_google_cloud_access = args.enable_google_cloud_access
    opts.binauthz_evaluation_mode = args.binauthz_evaluation_mode
    opts.binauthz_policy_bindings = args.binauthz_policy_bindings
    opts.stack_type = args.stack_type
    opts.gateway_api = args.gateway_api
    opts.logging_variant = args.logging_variant
    opts.additional_pod_ipv4_ranges = args.additional_pod_ipv4_ranges
    opts.removed_additional_pod_ipv4_ranges = args.remove_additional_pod_ipv4_ranges
    opts.fleet_project = args.fleet_project
    opts.enable_fleet = args.enable_fleet
    opts.clear_fleet_project = args.clear_fleet_project
    opts.enable_security_posture = args.enable_security_posture
    opts.network_performance_config = args.network_performance_configs
    opts.enable_k8s_beta_apis = args.enable_kubernetes_unstable_apis
    opts.security_posture = args.security_posture
    opts.workload_vulnerability_scanning = args.workload_vulnerability_scanning
    opts.enable_runtime_vulnerability_insight = args.enable_runtime_vulnerability_insight
    opts.workload_policies = args.workload_policies
    opts.remove_workload_policies = args.remove_workload_policies
    opts.enable_fqdn_network_policy = args.enable_fqdn_network_policy
    opts.host_maintenance_interval = args.host_maintenance_interval
    opts.enable_multi_networking = args.enable_multi_networking
    opts.containerd_config_from_file = args.containerd_config_from_file
    opts.convert_to_autopilot = args.convert_to_autopilot
    opts.complete_convert_to_autopilot = args.complete_convert_to_autopilot
    opts.convert_to_standard = args.convert_to_standard
    opts.enable_secret_manager = args.enable_secret_manager
    opts.enable_cilium_clusterwide_network_policy = args.enable_cilium_clusterwide_network_policy
    opts.enable_insecure_kubelet_readonly_port = args.enable_insecure_kubelet_readonly_port
    opts.autoprovisioning_enable_insecure_kubelet_readonly_port = args.autoprovisioning_enable_insecure_kubelet_readonly_port
    return opts