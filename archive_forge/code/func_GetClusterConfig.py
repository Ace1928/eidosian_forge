from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from stdin.
def GetClusterConfig(args, dataproc, project_id, compute_resources, beta=False, alpha=False, include_deprecated=True, include_ttl_config=False, include_gke_platform_args=False):
    """Get dataproc cluster configuration.

  Args:
    args: Arguments parsed from argparse.ArgParser.
    dataproc: Dataproc object that contains client, messages, and resources
    project_id: Dataproc project ID
    compute_resources: compute resource for cluster
    beta: use BETA only features
    alpha: use ALPHA only features
    include_deprecated: whether to include deprecated args
    include_ttl_config: whether to include Scheduled Delete(TTL) args
    include_gke_platform_args: whether to include GKE-based cluster args

  Returns:
    cluster_config: Dataproc cluster configuration
  """
    master_accelerator_type = None
    worker_accelerator_type = None
    secondary_worker_accelerator_type = None
    driver_pool_accelerator_type = None
    if args.master_accelerator:
        if 'type' in args.master_accelerator.keys():
            master_accelerator_type = args.master_accelerator['type']
        else:
            raise exceptions.ArgumentError('master-accelerator missing type!')
        master_accelerator_count = args.master_accelerator.get('count', 1)
    if args.worker_accelerator:
        if 'type' in args.worker_accelerator.keys():
            worker_accelerator_type = args.worker_accelerator['type']
        else:
            raise exceptions.ArgumentError('worker-accelerator missing type!')
        worker_accelerator_count = args.worker_accelerator.get('count', 1)
    secondary_worker_accelerator = _FirstNonNone(args.secondary_worker_accelerator, args.preemptible_worker_accelerator)
    if secondary_worker_accelerator:
        if 'type' in secondary_worker_accelerator.keys():
            secondary_worker_accelerator_type = secondary_worker_accelerator['type']
        else:
            raise exceptions.ArgumentError('secondary-worker-accelerator missing type!')
        secondary_worker_accelerator_count = secondary_worker_accelerator.get('count', 1)
    if args.min_worker_fraction and (args.min_worker_fraction < 0 or args.min_worker_fraction > 1):
        raise exceptions.ArgumentError('--min-worker-fraction must be between 0 and 1')
    if hasattr(args, 'driver_pool_accelerator') and args.driver_pool_accelerator:
        if 'type' in args.driver_pool_accelerator.keys():
            driver_pool_accelerator_type = args.driver_pool_accelerator['type']
        else:
            raise exceptions.ArgumentError('driver-pool-accelerator missing type!')
        driver_pool_accelerator_count = args.driver_pool_accelerator.get('count', 1)
    image_ref = args.image and compute_resources.Parse(args.image, params={'project': project_id}, collection='compute.images')
    network_ref = args.network and compute_resources.Parse(args.network, params={'project': project_id}, collection='compute.networks')
    subnetwork_ref = args.subnet and compute_resources.Parse(args.subnet, params={'project': project_id, 'region': properties.VALUES.compute.region.GetOrFail}, collection='compute.subnetworks')
    timeout_str = six.text_type(args.initialization_action_timeout) + 's'
    init_actions = [dataproc.messages.NodeInitializationAction(executableFile=exe, executionTimeout=timeout_str) for exe in args.initialization_actions or []]
    args.timeout += args.initialization_action_timeout * len(init_actions)
    expanded_scopes = compute_helpers.ExpandScopeAliases(args.scopes)
    software_config = dataproc.messages.SoftwareConfig(imageVersion=args.image_version)
    if include_deprecated:
        master_boot_disk_size_gb = args.master_boot_disk_size_gb
    else:
        master_boot_disk_size_gb = None
    if args.master_boot_disk_size:
        master_boot_disk_size_gb = api_utils.BytesToGb(args.master_boot_disk_size)
    if include_deprecated:
        worker_boot_disk_size_gb = args.worker_boot_disk_size_gb
    else:
        worker_boot_disk_size_gb = None
    if args.worker_boot_disk_size:
        worker_boot_disk_size_gb = api_utils.BytesToGb(args.worker_boot_disk_size)
    secondary_worker_boot_disk_size_gb = api_utils.BytesToGb(_FirstNonNone(args.secondary_worker_boot_disk_size, args.preemptible_worker_boot_disk_size))
    driver_pool_boot_disk_size_gb = None
    if hasattr(args, 'driver_pool_boot_disk_size') and args.driver_pool_boot_disk_size:
        driver_pool_boot_disk_size_gb = api_utils.BytesToGb(args.driver_pool_boot_disk_size)
    if args.single_node or args.num_workers == 0:
        args.properties[constants.ALLOW_ZERO_WORKERS_PROPERTY] = 'true'
    if args.enable_node_groups is not None:
        args.properties[constants.ENABLE_NODE_GROUPS_PROPERTY] = str(args.enable_node_groups).lower()
    if args.properties:
        software_config.properties = encoding.DictToAdditionalPropertyMessage(args.properties, dataproc.messages.SoftwareConfig.PropertiesValue, sort_items=True)
    if args.components:
        software_config_cls = dataproc.messages.SoftwareConfig
        software_config.optionalComponents.extend(list(map(software_config_cls.OptionalComponentsValueListEntryValuesEnum, args.components)))
    gce_cluster_config = dataproc.messages.GceClusterConfig(networkUri=network_ref and network_ref.SelfLink(), subnetworkUri=subnetwork_ref and subnetwork_ref.SelfLink(), privateIpv6GoogleAccess=_GetPrivateIpv6GoogleAccess(dataproc, args.private_ipv6_google_access_type), serviceAccount=args.service_account, serviceAccountScopes=expanded_scopes, zoneUri=properties.VALUES.compute.zone.GetOrFail())
    if args.public_ip_address:
        gce_cluster_config.internalIpOnly = not args.public_ip_address
    if args.no_address:
        gce_cluster_config.internalIpOnly = args.no_address
    reservation_affinity = GetReservationAffinity(args, dataproc)
    gce_cluster_config.reservationAffinity = reservation_affinity
    if args.tags:
        gce_cluster_config.tags = args.tags
    if args.metadata:
        flat_metadata = collections.OrderedDict()
        for entry in args.metadata:
            for k, v in entry.items():
                flat_metadata[k] = v
        gce_cluster_config.metadata = encoding.DictToAdditionalPropertyMessage(flat_metadata, dataproc.messages.GceClusterConfig.MetadataValue)
    master_accelerators = []
    if master_accelerator_type:
        master_accelerators.append(dataproc.messages.AcceleratorConfig(acceleratorTypeUri=master_accelerator_type, acceleratorCount=master_accelerator_count))
    worker_accelerators = []
    if worker_accelerator_type:
        worker_accelerators.append(dataproc.messages.AcceleratorConfig(acceleratorTypeUri=worker_accelerator_type, acceleratorCount=worker_accelerator_count))
    secondary_worker_accelerators = []
    if secondary_worker_accelerator_type:
        secondary_worker_accelerators.append(dataproc.messages.AcceleratorConfig(acceleratorTypeUri=secondary_worker_accelerator_type, acceleratorCount=secondary_worker_accelerator_count))
    driver_pool_accelerators = []
    if driver_pool_accelerator_type:
        driver_pool_accelerators.append(dataproc.messages.AcceleratorConfig(acceleratorTypeUri=driver_pool_accelerator_type, acceleratorCount=driver_pool_accelerator_count))
    cluster_config = dataproc.messages.ClusterConfig(configBucket=args.bucket, tempBucket=args.temp_bucket, gceClusterConfig=gce_cluster_config, masterConfig=dataproc.messages.InstanceGroupConfig(numInstances=args.num_masters, imageUri=image_ref and image_ref.SelfLink(), machineTypeUri=args.master_machine_type, accelerators=master_accelerators, diskConfig=GetDiskConfig(dataproc, args.master_boot_disk_type, master_boot_disk_size_gb, args.num_master_local_ssds, args.master_local_ssd_interface), minCpuPlatform=args.master_min_cpu_platform), workerConfig=dataproc.messages.InstanceGroupConfig(numInstances=args.num_workers, minNumInstances=args.min_num_workers, imageUri=image_ref and image_ref.SelfLink(), machineTypeUri=args.worker_machine_type, accelerators=worker_accelerators, diskConfig=GetDiskConfig(dataproc, args.worker_boot_disk_type, worker_boot_disk_size_gb, args.num_worker_local_ssds, args.worker_local_ssd_interface), minCpuPlatform=args.worker_min_cpu_platform), initializationActions=init_actions, softwareConfig=software_config)
    if args.min_worker_fraction:
        cluster_config.workerConfig.startupConfig = dataproc.messages.StartupConfig(requiredRegistrationFraction=args.min_worker_fraction)
    if args.kerberos_config_file or args.enable_kerberos or args.kerberos_root_principal_password_uri:
        cluster_config.securityConfig = dataproc.messages.SecurityConfig()
        if args.kerberos_config_file:
            cluster_config.securityConfig.kerberosConfig = ParseKerberosConfigFile(dataproc, args.kerberos_config_file)
        else:
            kerberos_config = dataproc.messages.KerberosConfig()
            if args.enable_kerberos:
                kerberos_config.enableKerberos = args.enable_kerberos
            else:
                kerberos_config.enableKerberos = True
            if args.kerberos_root_principal_password_uri:
                kerberos_config.rootPrincipalPasswordUri = args.kerberos_root_principal_password_uri
                kerberos_kms_ref = args.CONCEPTS.kerberos_kms_key.Parse()
                if kerberos_kms_ref:
                    kerberos_config.kmsKeyUri = kerberos_kms_ref.RelativeName()
            cluster_config.securityConfig.kerberosConfig = kerberos_config
    if not beta:
        if args.identity_config_file or args.secure_multi_tenancy_user_mapping:
            if cluster_config.securityConfig is None:
                cluster_config.securityConfig = dataproc.messages.SecurityConfig()
            if args.identity_config_file:
                cluster_config.securityConfig.identityConfig = ParseIdentityConfigFile(dataproc, args.identity_config_file)
            else:
                user_service_account_mapping = ParseSecureMultiTenancyUserServiceAccountMappingString(args.secure_multi_tenancy_user_mapping)
                identity_config = dataproc.messages.IdentityConfig()
                identity_config.userServiceAccountMapping = encoding.DictToAdditionalPropertyMessage(user_service_account_mapping, dataproc.messages.IdentityConfig.UserServiceAccountMappingValue)
                cluster_config.securityConfig.identityConfig = identity_config
    if args.autoscaling_policy:
        cluster_config.autoscalingConfig = dataproc.messages.AutoscalingConfig(policyUri=args.CONCEPTS.autoscaling_policy.Parse().RelativeName())
    if args.node_group:
        gce_cluster_config.nodeGroupAffinity = dataproc.messages.NodeGroupAffinity(nodeGroupUri=args.node_group)
    if args.IsSpecified('shielded_secure_boot') or args.IsSpecified('shielded_vtpm') or args.IsSpecified('shielded_integrity_monitoring'):
        gce_cluster_config.shieldedInstanceConfig = dataproc.messages.ShieldedInstanceConfig(enableSecureBoot=args.shielded_secure_boot, enableVtpm=args.shielded_vtpm, enableIntegrityMonitoring=args.shielded_integrity_monitoring)
    if not beta and args.IsSpecified('confidential_compute'):
        gce_cluster_config.confidentialInstanceConfig = dataproc.messages.ConfidentialInstanceConfig(enableConfidentialCompute=args.confidential_compute)
    if args.dataproc_metastore:
        cluster_config.metastoreConfig = dataproc.messages.MetastoreConfig(dataprocMetastoreService=args.dataproc_metastore)
    if include_ttl_config:
        lifecycle_config = dataproc.messages.LifecycleConfig()
        changed_config = False
        if args.max_age is not None:
            lifecycle_config.autoDeleteTtl = six.text_type(args.max_age) + 's'
            changed_config = True
        if args.expiration_time is not None:
            lifecycle_config.autoDeleteTime = times.FormatDateTime(args.expiration_time)
            changed_config = True
        if args.max_idle is not None:
            lifecycle_config.idleDeleteTtl = six.text_type(args.max_idle) + 's'
            changed_config = True
        if changed_config:
            cluster_config.lifecycleConfig = lifecycle_config
    encryption_config = dataproc.messages.EncryptionConfig()
    if hasattr(args.CONCEPTS, 'gce_pd_kms_key'):
        gce_pd_kms_ref = args.CONCEPTS.gce_pd_kms_key.Parse()
        if gce_pd_kms_ref:
            encryption_config.gcePdKmsKeyName = gce_pd_kms_ref.RelativeName()
        else:
            for keyword in ['gce-pd-kms-key', 'gce-pd-kms-key-project', 'gce-pd-kms-key-location', 'gce-pd-kms-key-keyring']:
                if getattr(args, keyword.replace('-', '_'), None):
                    raise exceptions.ArgumentError('--gce-pd-kms-key was not fully specified.')
    if hasattr(args.CONCEPTS, 'kms_key'):
        kms_ref = args.CONCEPTS.kms_key.Parse()
        if kms_ref:
            encryption_config.kmsKey = kms_ref.RelativeName()
        else:
            for keyword in ['kms-key', 'kms-project', 'kms-location', 'kms-keyring']:
                if getattr(args, keyword.replace('-', '_'), None):
                    raise exceptions.ArgumentError('--kms-key was not fully specified.')
    if encryption_config.gcePdKmsKeyName or encryption_config.kmsKey:
        cluster_config.encryptionConfig = encryption_config
    num_secondary_workers = _FirstNonNone(args.num_secondary_workers, args.num_preemptible_workers)
    secondary_worker_boot_disk_type = _FirstNonNone(args.secondary_worker_boot_disk_type, args.preemptible_worker_boot_disk_type)
    num_secondary_worker_local_ssds = _FirstNonNone(args.num_secondary_worker_local_ssds, args.num_preemptible_worker_local_ssds)
    if num_secondary_workers is not None or secondary_worker_boot_disk_size_gb is not None or secondary_worker_boot_disk_type is not None or (num_secondary_worker_local_ssds is not None) or (args.worker_min_cpu_platform is not None) or (args.secondary_worker_type == 'non-preemptible') or (args.secondary_worker_type == 'spot') or (args.secondary_worker_machine_types is not None) or (args.min_secondary_worker_fraction is not None):
        instance_flexibility_policy = GetInstanceFlexibilityPolicy(dataproc, args, alpha)
        startup_config = GetStartupConfig(dataproc, args)
        cluster_config.secondaryWorkerConfig = dataproc.messages.InstanceGroupConfig(numInstances=num_secondary_workers, accelerators=secondary_worker_accelerators, diskConfig=GetDiskConfig(dataproc, secondary_worker_boot_disk_type, secondary_worker_boot_disk_size_gb, num_secondary_worker_local_ssds, args.secondary_worker_local_ssd_interface), minCpuPlatform=args.worker_min_cpu_platform, preemptibility=_GetInstanceGroupPreemptibility(dataproc, args.secondary_worker_type), instanceFlexibilityPolicy=instance_flexibility_policy, startupConfig=startup_config)
    if _AtLeastOneGceNodePoolSpecified(args, driver_pool_boot_disk_size_gb):
        cluster_config.auxiliaryNodeGroups = [dataproc.messages.AuxiliaryNodeGroup(nodeGroup=dataproc.messages.NodeGroup(labels=labels_util.ParseCreateArgs(args, dataproc.messages.NodeGroup.LabelsValue), roles=[dataproc.messages.NodeGroup.RolesValueListEntryValuesEnum.DRIVER], nodeGroupConfig=dataproc.messages.InstanceGroupConfig(numInstances=args.driver_pool_size, imageUri=image_ref and image_ref.SelfLink(), machineTypeUri=args.driver_pool_machine_type, accelerators=driver_pool_accelerators, diskConfig=GetDiskConfig(dataproc, args.driver_pool_boot_disk_type, driver_pool_boot_disk_size_gb, args.num_driver_pool_local_ssds, args.driver_pool_local_ssd_interface), minCpuPlatform=args.driver_pool_min_cpu_platform)), nodeGroupId=args.driver_pool_id)]
    if args.enable_component_gateway:
        cluster_config.endpointConfig = dataproc.messages.EndpointConfig(enableHttpPortAccess=args.enable_component_gateway)
    if include_gke_platform_args:
        if args.gke_cluster is not None:
            location = args.zone or args.region
            target_gke_cluster = 'projects/{0}/locations/{1}/clusters/{2}'.format(project_id, location, args.gke_cluster)
            cluster_config.gkeClusterConfig = dataproc.messages.GkeClusterConfig(namespacedGkeDeploymentTarget=dataproc.messages.NamespacedGkeDeploymentTarget(targetGkeCluster=target_gke_cluster, clusterNamespace=args.gke_cluster_namespace))
            cluster_config.gceClusterConfig = None
            cluster_config.masterConfig = None
            cluster_config.workerConfig = None
            cluster_config.secondaryWorkerConfig = None
    if not beta and args.metric_sources:
        _SetDataprocMetricConfig(args, cluster_config, dataproc)
    return cluster_config