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
def UpdateClusterCommon(self, cluster_ref, options):
    """Returns an UpdateCluster operation."""
    update = None
    if not options.version:
        options.version = '-'
    if options.update_nodes:
        update = self.messages.ClusterUpdate(desiredNodeVersion=options.version, desiredNodePoolId=options.node_pool, desiredImageType=options.image_type, desiredImage=options.image, desiredImageProject=options.image_project)
        if options.security_profile is not None:
            update.securityProfile = self.messages.SecurityProfile(name=options.security_profile)
    elif options.update_master:
        update = self.messages.ClusterUpdate(desiredMasterVersion=options.version)
        if options.security_profile is not None:
            update.securityProfile = self.messages.SecurityProfile(name=options.security_profile)
    elif options.enable_stackdriver_kubernetes:
        update = self.messages.ClusterUpdate()
        update.desiredLoggingService = 'logging.googleapis.com/kubernetes'
        update.desiredMonitoringService = 'monitoring.googleapis.com/kubernetes'
    elif options.enable_stackdriver_kubernetes is not None:
        update = self.messages.ClusterUpdate()
        update.desiredLoggingService = 'none'
        update.desiredMonitoringService = 'none'
    elif options.monitoring_service or options.logging_service:
        update = self.messages.ClusterUpdate()
        if options.monitoring_service:
            update.desiredMonitoringService = options.monitoring_service
        if options.logging_service:
            update.desiredLoggingService = options.logging_service
    elif options.logging or options.monitoring or options.enable_managed_prometheus or options.disable_managed_prometheus or options.enable_dataplane_v2_metrics or options.disable_dataplane_v2_metrics or options.enable_dataplane_v2_flow_observability or options.disable_dataplane_v2_flow_observability or options.dataplane_v2_observability_mode:
        logging = _GetLoggingConfig(options, self.messages)
        if (options.dataplane_v2_observability_mode or options.enable_dataplane_v2_flow_observability or options.disable_dataplane_v2_flow_observability) and options.enable_dataplane_v2_metrics is None and (options.disable_dataplane_v2_metrics is None):
            cluster = self.GetCluster(cluster_ref)
            if cluster and cluster.monitoringConfig and cluster.monitoringConfig.advancedDatapathObservabilityConfig:
                if cluster.monitoringConfig.advancedDatapathObservabilityConfig.enableMetrics:
                    options.enable_dataplane_v2_metrics = True
                else:
                    options.disable_dataplane_v2_metrics = True
        monitoring = _GetMonitoringConfig(options, self.messages)
        update = self.messages.ClusterUpdate()
        if logging:
            update.desiredLoggingConfig = logging
        if monitoring:
            update.desiredMonitoringConfig = monitoring
    elif options.disable_addons:
        disable_node_local_dns = options.disable_addons.get(NODELOCALDNS)
        addons = self._AddonsConfig(disable_ingress=options.disable_addons.get(INGRESS), disable_hpa=options.disable_addons.get(HPA), disable_dashboard=options.disable_addons.get(DASHBOARD), disable_network_policy=options.disable_addons.get(NETWORK_POLICY), enable_node_local_dns=not disable_node_local_dns if disable_node_local_dns is not None else None)
        if options.disable_addons.get(CONFIGCONNECTOR) is not None:
            addons.configConnectorConfig = self.messages.ConfigConnectorConfig(enabled=not options.disable_addons.get(CONFIGCONNECTOR))
        if options.disable_addons.get(GCEPDCSIDRIVER) is not None:
            addons.gcePersistentDiskCsiDriverConfig = self.messages.GcePersistentDiskCsiDriverConfig(enabled=not options.disable_addons.get(GCEPDCSIDRIVER))
        if options.disable_addons.get(GCPFILESTORECSIDRIVER) is not None:
            addons.gcpFilestoreCsiDriverConfig = self.messages.GcpFilestoreCsiDriverConfig(enabled=not options.disable_addons.get(GCPFILESTORECSIDRIVER))
        if options.disable_addons.get(GCSFUSECSIDRIVER) is not None:
            addons.gcsFuseCsiDriverConfig = self.messages.GcsFuseCsiDriverConfig(enabled=not options.disable_addons.get(GCSFUSECSIDRIVER))
        if options.disable_addons.get(STATEFULHA) is not None:
            addons.statefulHaConfig = self.messages.StatefulHAConfig(enabled=not options.disable_addons.get(STATEFULHA))
        if options.disable_addons.get(PARALLELSTORECSIDRIVER) is not None:
            addons.parallelstoreCsiDriverConfig = self.messages.ParallelstoreCsiDriverConfig(enabled=not options.disable_addons.get(PARALLELSTORECSIDRIVER))
        if options.disable_addons.get(BACKUPRESTORE) is not None:
            addons.gkeBackupAgentConfig = self.messages.GkeBackupAgentConfig(enabled=not options.disable_addons.get(BACKUPRESTORE))
        update = self.messages.ClusterUpdate(desiredAddonsConfig=addons)
    elif options.enable_autoscaling is not None:
        autoscaling = self.messages.NodePoolAutoscaling(enabled=options.enable_autoscaling)
        if options.enable_autoscaling:
            autoscaling.minNodeCount = options.min_nodes
            autoscaling.maxNodeCount = options.max_nodes
            autoscaling.totalMinNodeCount = options.total_min_nodes
            autoscaling.totalMaxNodeCount = options.total_max_nodes
            if options.location_policy is not None:
                autoscaling.locationPolicy = LocationPolicyEnumFromString(self.messages, options.location_policy)
        update = self.messages.ClusterUpdate(desiredNodePoolId=options.node_pool, desiredNodePoolAutoscaling=autoscaling)
    elif options.locations:
        update = self.messages.ClusterUpdate(desiredLocations=options.locations)
    elif options.enable_master_authorized_networks is not None:
        authorized_networks = self.messages.MasterAuthorizedNetworksConfig(enabled=options.enable_master_authorized_networks)
        if options.master_authorized_networks:
            for network in options.master_authorized_networks:
                authorized_networks.cidrBlocks.append(self.messages.CidrBlock(cidrBlock=network))
        update = self.messages.ClusterUpdate(desiredMasterAuthorizedNetworksConfig=authorized_networks)
    elif options.enable_autoprovisioning is not None or options.autoscaling_profile is not None:
        autoscaling = self.CreateClusterAutoscalingCommon(cluster_ref, options, True)
        update = self.messages.ClusterUpdate(desiredClusterAutoscaling=autoscaling)
    elif options.enable_pod_security_policy is not None:
        config = self.messages.PodSecurityPolicyConfig(enabled=options.enable_pod_security_policy)
        update = self.messages.ClusterUpdate(desiredPodSecurityPolicyConfig=config)
    elif options.enable_vertical_pod_autoscaling is not None:
        vertical_pod_autoscaling = self.messages.VerticalPodAutoscaling(enabled=options.enable_vertical_pod_autoscaling)
        update = self.messages.ClusterUpdate(desiredVerticalPodAutoscaling=vertical_pod_autoscaling)
    elif options.resource_usage_bigquery_dataset is not None:
        export_config = self.messages.ResourceUsageExportConfig(bigqueryDestination=self.messages.BigQueryDestination(datasetId=options.resource_usage_bigquery_dataset))
        if options.enable_network_egress_metering:
            export_config.enableNetworkEgressMetering = True
        if options.enable_resource_consumption_metering is not None:
            export_config.consumptionMeteringConfig = self.messages.ConsumptionMeteringConfig(enabled=options.enable_resource_consumption_metering)
        update = self.messages.ClusterUpdate(desiredResourceUsageExportConfig=export_config)
    elif options.enable_network_egress_metering is not None:
        raise util.Error(ENABLE_NETWORK_EGRESS_METERING_ERROR_MSG)
    elif options.enable_resource_consumption_metering is not None:
        raise util.Error(ENABLE_RESOURCE_CONSUMPTION_METERING_ERROR_MSG)
    elif options.clear_resource_usage_bigquery_dataset is not None:
        export_config = self.messages.ResourceUsageExportConfig()
        update = self.messages.ClusterUpdate(desiredResourceUsageExportConfig=export_config)
    elif options.security_profile is not None:
        security_profile = self.messages.SecurityProfile(name=options.security_profile)
        update = self.messages.ClusterUpdate(securityProfile=security_profile)
    elif options.enable_intra_node_visibility is not None:
        intra_node_visibility_config = self.messages.IntraNodeVisibilityConfig(enabled=options.enable_intra_node_visibility)
        update = self.messages.ClusterUpdate(desiredIntraNodeVisibilityConfig=intra_node_visibility_config)
    elif options.enable_master_global_access is not None:
        master_global_access_config = self.messages.PrivateClusterMasterGlobalAccessConfig(enabled=options.enable_master_global_access)
        private_cluster_config = self.messages.PrivateClusterConfig(masterGlobalAccessConfig=master_global_access_config)
        update = self.messages.ClusterUpdate(desiredPrivateClusterConfig=private_cluster_config)
    if options.security_profile is not None and options.security_profile_runtime_rules is not None:
        update.securityProfile.disableRuntimeRules = not options.security_profile_runtime_rules
    if options.master_authorized_networks and (not options.enable_master_authorized_networks):
        raise util.Error(MISMATCH_AUTHORIZED_NETWORKS_ERROR_MSG)
    if options.database_encryption_key:
        update = self.messages.ClusterUpdate(desiredDatabaseEncryption=self.messages.DatabaseEncryption(keyName=options.database_encryption_key, state=self.messages.DatabaseEncryption.StateValueValuesEnum.ENCRYPTED))
    elif options.disable_database_encryption:
        update = self.messages.ClusterUpdate(desiredDatabaseEncryption=self.messages.DatabaseEncryption(state=self.messages.DatabaseEncryption.StateValueValuesEnum.DECRYPTED))
    if options.enable_shielded_nodes is not None:
        update = self.messages.ClusterUpdate(desiredShieldedNodes=self.messages.ShieldedNodes(enabled=options.enable_shielded_nodes))
    if options.enable_tpu is not None:
        update = self.messages.ClusterUpdate(desiredTpuConfig=_GetTpuConfigForClusterUpdate(options, self.messages))
    if options.release_channel is not None:
        update = self.messages.ClusterUpdate(desiredReleaseChannel=_GetReleaseChannel(options, self.messages))
    if options.disable_default_snat is not None:
        disable_default_snat = self.messages.DefaultSnatStatus(disabled=options.disable_default_snat)
        update = self.messages.ClusterUpdate(desiredDefaultSnatStatus=disable_default_snat)
    if options.enable_l4_ilb_subsetting is not None:
        ilb_subsettting_config = self.messages.ILBSubsettingConfig(enabled=options.enable_l4_ilb_subsetting)
        update = self.messages.ClusterUpdate(desiredL4ilbSubsettingConfig=ilb_subsettting_config)
    if options.private_ipv6_google_access_type is not None:
        update = self.messages.ClusterUpdate(desiredPrivateIpv6GoogleAccess=util.GetPrivateIpv6GoogleAccessTypeMapperForUpdate(self.messages, hidden=False).GetEnumForChoice(options.private_ipv6_google_access_type))
    dns_config = self.ParseClusterDNSOptions(options, is_update=True)
    if dns_config is not None:
        update = self.messages.ClusterUpdate(desiredDnsConfig=dns_config)
    gateway_config = self.ParseGatewayOptions(options)
    if gateway_config is not None:
        update = self.messages.ClusterUpdate(desiredGatewayApiConfig=gateway_config)
    if options.notification_config is not None:
        update = self.messages.ClusterUpdate(desiredNotificationConfig=_GetNotificationConfigForClusterUpdate(options, self.messages))
    if options.disable_autopilot is not None:
        update = self.messages.ClusterUpdate(desiredAutopilot=self.messages.Autopilot(enabled=False))
    if options.security_group is not None:
        update = self.messages.ClusterUpdate(desiredAuthenticatorGroupsConfig=self.messages.AuthenticatorGroupsConfig(enabled=True, securityGroup=options.security_group))
    if options.enable_gcfs is not None:
        update = self.messages.ClusterUpdate(desiredGcfsConfig=self.messages.GcfsConfig(enabled=options.enable_gcfs))
    if options.autoprovisioning_network_tags is not None:
        update = self.messages.ClusterUpdate(desiredNodePoolAutoConfigNetworkTags=self.messages.NetworkTags(tags=options.autoprovisioning_network_tags))
    if options.autoprovisioning_resource_manager_tags is not None:
        tags = options.autoprovisioning_resource_manager_tags
        rm_tags = self._ResourceManagerTags(tags)
        update = self.messages.ClusterUpdate(desiredNodePoolAutoConfigResourceManagerTags=rm_tags)
    if options.enable_image_streaming is not None:
        update = self.messages.ClusterUpdate(desiredGcfsConfig=self.messages.GcfsConfig(enabled=options.enable_image_streaming))
    if options.enable_mesh_certificates is not None:
        update = self.messages.ClusterUpdate(desiredMeshCertificates=self.messages.MeshCertificates(enableCertificates=options.enable_mesh_certificates))
    if options.maintenance_interval is not None:
        update = self.messages.ClusterUpdate(desiredStableFleetConfig=_GetStableFleetConfig(options, self.messages))
    if options.enable_service_externalips is not None:
        update = self.messages.ClusterUpdate(desiredServiceExternalIpsConfig=self.messages.ServiceExternalIPsConfig(enabled=options.enable_service_externalips))
    if options.enable_identity_service is not None:
        update = self.messages.ClusterUpdate(desiredIdentityServiceConfig=self.messages.IdentityServiceConfig(enabled=options.enable_identity_service))
    if options.enable_workload_config_audit is not None or options.enable_workload_vulnerability_scanning is not None:
        protect_config = self.messages.ProtectConfig()
        if options.enable_workload_config_audit is not None:
            protect_config.workloadConfig = self.messages.WorkloadConfig()
            if options.enable_workload_config_audit:
                protect_config.workloadConfig.auditMode = self.messages.WorkloadConfig.AuditModeValueValuesEnum.BASIC
            else:
                protect_config.workloadConfig.auditMode = self.messages.WorkloadConfig.AuditModeValueValuesEnum.DISABLED
        if options.enable_workload_vulnerability_scanning is not None:
            if options.enable_workload_vulnerability_scanning:
                protect_config.workloadVulnerabilityMode = self.messages.ProtectConfig.WorkloadVulnerabilityModeValueValuesEnum.BASIC
            else:
                protect_config.workloadVulnerabilityMode = self.messages.ProtectConfig.WorkloadVulnerabilityModeValueValuesEnum.DISABLED
        update = self.messages.ClusterUpdate(desiredProtectConfig=protect_config)
    if options.pod_autoscaling_direct_metrics_opt_in is not None:
        pod_autoscaling_config = self.messages.PodAutoscaling(directMetricsOptIn=options.pod_autoscaling_direct_metrics_opt_in)
        update = self.messages.ClusterUpdate(desiredPodAutoscaling=pod_autoscaling_config)
    if options.enable_private_endpoint is not None:
        update = self.messages.ClusterUpdate(desiredEnablePrivateEndpoint=options.enable_private_endpoint)
    if options.logging_variant is not None:
        logging_config = self.messages.NodePoolLoggingConfig()
        logging_config.variantConfig = self.messages.LoggingVariantConfig(variant=VariantConfigEnumFromString(self.messages, options.logging_variant))
        update = self.messages.ClusterUpdate(desiredNodePoolLoggingConfig=logging_config)
    if options.additional_pod_ipv4_ranges or options.removed_additional_pod_ipv4_ranges:
        update = self.messages.ClusterUpdate()
        if options.additional_pod_ipv4_ranges:
            update.additionalPodRangesConfig = self.messages.AdditionalPodRangesConfig(podRangeNames=options.additional_pod_ipv4_ranges)
        if options.removed_additional_pod_ipv4_ranges:
            update.removedAdditionalPodRangesConfig = self.messages.AdditionalPodRangesConfig(podRangeNames=options.removed_additional_pod_ipv4_ranges)
    if options.stack_type is not None:
        update = self.messages.ClusterUpdate(desiredStackType=util.GetUpdateStackTypeMapper(self.messages).GetEnumForChoice(options.stack_type))
    if options.enable_cost_allocation is not None:
        update = self.messages.ClusterUpdate(desiredCostManagementConfig=self.messages.CostManagementConfig(enabled=options.enable_cost_allocation))
    if options.enable_fleet:
        update = self.messages.ClusterUpdate(desiredFleet=self.messages.Fleet(project=cluster_ref.projectId))
    if options.fleet_project:
        update = self.messages.ClusterUpdate(desiredFleet=self.messages.Fleet(project=options.fleet_project))
    if options.enable_k8s_beta_apis is not None:
        config_obj = self.messages.K8sBetaAPIConfig()
        config_obj.enabledApis = options.enable_k8s_beta_apis
        update = self.messages.ClusterUpdate(desiredK8sBetaApis=config_obj)
    if options.clear_fleet_project:
        update = self.messages.ClusterUpdate(desiredFleet=self.messages.Fleet(project=''))
    if options.enable_security_posture is not None:
        security_posture_config = self.messages.SecurityPostureConfig()
        if options.enable_security_posture:
            security_posture_config.mode = self.messages.SecurityPostureConfig.ModeValueValuesEnum.BASIC
        else:
            security_posture_config.mode = self.messages.SecurityPostureConfig.ModeValueValuesEnum.DISABLED
        update = self.messages.ClusterUpdate(desiredSecurityPostureConfig=security_posture_config)
    if options.security_posture is not None:
        security_posture_config = self.messages.SecurityPostureConfig()
        if options.security_posture.lower() == 'enterprise':
            security_posture_config.mode = self.messages.SecurityPostureConfig.ModeValueValuesEnum.ENTERPRISE
        elif options.security_posture.lower() == 'standard':
            security_posture_config.mode = self.messages.SecurityPostureConfig.ModeValueValuesEnum.BASIC
        elif options.security_posture.lower() == 'disabled':
            security_posture_config.mode = self.messages.SecurityPostureConfig.ModeValueValuesEnum.DISABLED
        else:
            raise util.Error(SECURITY_POSTURE_MODE_NOT_SUPPORTED.format(mode=options.security_posture.lower()))
        update = self.messages.ClusterUpdate(desiredSecurityPostureConfig=security_posture_config)
    if options.workload_vulnerability_scanning is not None:
        security_posture_config = self.messages.SecurityPostureConfig()
        if options.workload_vulnerability_scanning.lower() == 'standard':
            security_posture_config.vulnerabilityMode = self.messages.SecurityPostureConfig.VulnerabilityModeValueValuesEnum.VULNERABILITY_BASIC
        elif options.workload_vulnerability_scanning.lower() == 'disabled':
            security_posture_config.vulnerabilityMode = self.messages.SecurityPostureConfig.VulnerabilityModeValueValuesEnum.VULNERABILITY_DISABLED
        elif options.workload_vulnerability_scanning.lower() == 'enterprise':
            security_posture_config.vulnerabilityMode = self.messages.SecurityPostureConfig.VulnerabilityModeValueValuesEnum.VULNERABILITY_ENTERPRISE
        else:
            raise util.Error(WORKLOAD_VULNERABILITY_SCANNING_MODE_NOT_SUPPORTED.format(mode=options.workload_vulnerability_scanning.lower()))
        update = self.messages.ClusterUpdate(desiredSecurityPostureConfig=security_posture_config)
    if options.enable_runtime_vulnerability_insight is not None:
        runtime_vulnerability_insight_config = self.messages.RuntimeVulnerabilityInsightConfig()
        if options.enable_runtime_vulnerability_insight:
            runtime_vulnerability_insight_config.mode = self.messages.RuntimeVulnerabilityInsightConfig.ModeValueValuesEnum.PREMIUM_VULNERABILITY_SCAN
        else:
            runtime_vulnerability_insight_config.mode = self.messages.RuntimeVulnerabilityInsightConfig.ModeValueValuesEnum.DISABLED
        update = self.messages.ClusterUpdate(desiredRuntimeVulnerabilityInsightConfig=runtime_vulnerability_insight_config)
    if options.network_performance_config:
        perf = self._GetClusterNetworkPerformanceConfig(options)
        update = self.messages.ClusterUpdate(desiredNetworkPerformanceConfig=perf)
    if options.workload_policies is not None:
        workload_policies = self.messages.WorkloadPolicyConfig()
        if options.workload_policies == 'allow-net-admin':
            workload_policies.allowNetAdmin = True
        update = self.messages.ClusterUpdate(desiredAutopilotWorkloadPolicyConfig=workload_policies)
    if options.remove_workload_policies is not None:
        workload_policies = self.messages.WorkloadPolicyConfig()
        if options.remove_workload_policies == 'allow-net-admin':
            workload_policies.allowNetAdmin = False
        update = self.messages.ClusterUpdate(desiredAutopilotWorkloadPolicyConfig=workload_policies)
    if options.host_maintenance_interval is not None:
        update = self.messages.ClusterUpdate(desiredHostMaintenancePolicy=_GetHostMaintenancePolicy(options, self.messages))
    if options.in_transit_encryption is not None:
        update = self.messages.ClusterUpdate(desiredInTransitEncryptionConfig=util.GetUpdateInTransitEncryptionConfigMapper(self.messages).GetEnumForChoice(options.in_transit_encryption))
    if options.enable_multi_networking is not None:
        update = self.messages.ClusterUpdate(desiredEnableMultiNetworking=options.enable_multi_networking)
    if options.containerd_config_from_file is not None:
        update = self.messages.ClusterUpdate(desiredContainerdConfig=self.messages.ContainerdConfig())
        util.LoadContainerdConfigFromYAML(update.desiredContainerdConfig, options.containerd_config_from_file, self.messages)
    if options.enable_secret_manager is not None:
        update = self.messages.ClusterUpdate(desiredSecretManagerConfig=self.messages.SecretManagerConfig(enabled=options.enable_secret_manager))
    if options.enable_cilium_clusterwide_network_policy is not None:
        update = self.messages.ClusterUpdate(desiredEnableCiliumClusterwideNetworkPolicy=options.enable_cilium_clusterwide_network_policy)
    if options.enable_fqdn_network_policy is not None:
        update = self.messages.ClusterUpdate(desiredEnableFqdnNetworkPolicy=options.enable_fqdn_network_policy)
    return update