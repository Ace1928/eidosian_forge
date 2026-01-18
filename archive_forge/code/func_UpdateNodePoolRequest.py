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
def UpdateNodePoolRequest(self, node_pool_ref, options):
    """Creates an UpdateNodePoolRequest from the provided options.

    Arguments:
      node_pool_ref: The node pool to act on.
      options: UpdateNodePoolOptions with the user-specified options.

    Returns:

      An UpdateNodePoolRequest.
    """
    update_request = self.messages.UpdateNodePoolRequest(name=ProjectLocationClusterNodePool(node_pool_ref.projectId, node_pool_ref.zone, node_pool_ref.clusterId, node_pool_ref.nodePoolId))
    self.ParseAcceleratorOptions(options, update_request)
    if options.workload_metadata is not None or options.workload_metadata_from_node is not None:
        self._AddWorkloadMetadataToNodeConfig(update_request, options, self.messages)
    elif options.node_locations is not None:
        update_request.locations = sorted(options.node_locations)
    elif options.enable_blue_green_upgrade or options.enable_surge_upgrade or options.max_surge_upgrade is not None or (options.max_unavailable_upgrade is not None) or (options.standard_rollout_policy is not None) or (options.node_pool_soak_duration is not None) or options.autoscaled_rollout_policy:
        update_request.upgradeSettings = self.UpdateUpgradeSettings(node_pool_ref, options)
    elif options.system_config_from_file is not None or options.enable_insecure_kubelet_readonly_port is not None:
        node_config = self.messages.NodeConfig()
        if options.system_config_from_file is not None:
            util.LoadSystemConfigFromYAML(node_config, options.system_config_from_file, options.enable_insecure_kubelet_readonly_port, self.messages)
        if options.enable_insecure_kubelet_readonly_port is not None:
            if node_config.kubeletConfig is None:
                node_config.kubeletConfig = self.messages.NodeKubeletConfig()
            node_config.kubeletConfig.insecureKubeletReadonlyPortEnabled = options.enable_insecure_kubelet_readonly_port
        update_request.linuxNodeConfig = node_config.linuxNodeConfig
        update_request.kubeletConfig = node_config.kubeletConfig
    elif options.containerd_config_from_file is not None:
        containerd_config = self.messages.ContainerdConfig()
        util.LoadContainerdConfigFromYAML(containerd_config, options.containerd_config_from_file, self.messages)
        update_request.containerdConfig = containerd_config
    elif options.labels is not None:
        resource_labels = self.messages.ResourceLabels()
        labels = resource_labels.LabelsValue()
        props = []
        for key, value in six.iteritems(options.labels):
            props.append(labels.AdditionalProperty(key=key, value=value))
        labels.additionalProperties = props
        resource_labels.labels = labels
        update_request.resourceLabels = resource_labels
    elif options.node_labels is not None:
        node_labels = self.messages.NodeLabels()
        labels = node_labels.LabelsValue()
        props = []
        for key, value in six.iteritems(options.node_labels):
            props.append(labels.AdditionalProperty(key=key, value=value))
        labels.additionalProperties = props
        node_labels.labels = labels
        update_request.labels = node_labels
    elif options.node_taints is not None:
        taints = []
        effect_enum = self.messages.NodeTaint.EffectValueValuesEnum
        for key, value in sorted(six.iteritems(options.node_taints)):
            strs = value.split(':')
            if len(strs) != 2:
                raise util.Error(NODE_TAINT_INCORRECT_FORMAT_ERROR_MSG.format(key=key, value=value))
            value = strs[0]
            taint_effect = strs[1]
            if taint_effect == 'NoSchedule':
                effect = effect_enum.NO_SCHEDULE
            elif taint_effect == 'PreferNoSchedule':
                effect = effect_enum.PREFER_NO_SCHEDULE
            elif taint_effect == 'NoExecute':
                effect = effect_enum.NO_EXECUTE
            else:
                raise util.Error(NODE_TAINT_INCORRECT_EFFECT_ERROR_MSG.format(effect=strs[1]))
            taints.append(self.messages.NodeTaint(key=key, value=value, effect=effect))
        node_taints = self.messages.NodeTaints()
        node_taints.taints = taints
        update_request.taints = node_taints
    elif options.tags is not None:
        node_tags = self.messages.NetworkTags()
        node_tags.tags = options.tags
        update_request.tags = node_tags
    elif options.enable_private_nodes is not None:
        network_config = self.messages.NodeNetworkConfig()
        network_config.enablePrivateNodes = options.enable_private_nodes
        update_request.nodeNetworkConfig = network_config
    elif options.enable_gcfs is not None:
        gcfs_config = self.messages.GcfsConfig(enabled=options.enable_gcfs)
        update_request.gcfsConfig = gcfs_config
    elif options.gvnic is not None:
        gvnic = self.messages.VirtualNIC(enabled=options.gvnic)
        update_request.gvnic = gvnic
    elif options.enable_image_streaming is not None:
        gcfs_config = self.messages.GcfsConfig(enabled=options.enable_image_streaming)
        update_request.gcfsConfig = gcfs_config
    elif options.network_performance_config is not None:
        network_config = self.messages.NodeNetworkConfig()
        network_config.networkPerformanceConfig = self._GetNetworkPerformanceConfig(options)
        update_request.nodeNetworkConfig = network_config
    elif options.enable_confidential_nodes is not None:
        confidential_nodes = self.messages.ConfidentialNodes(enabled=options.enable_confidential_nodes)
        update_request.confidentialNodes = confidential_nodes
    elif options.enable_fast_socket is not None:
        fast_socket = self.messages.FastSocket(enabled=options.enable_fast_socket)
        update_request.fastSocket = fast_socket
    elif options.logging_variant is not None:
        logging_config = self.messages.NodePoolLoggingConfig()
        logging_config.variantConfig = self.messages.LoggingVariantConfig(variant=VariantConfigEnumFromString(self.messages, options.logging_variant))
        update_request.loggingConfig = logging_config
    elif options.windows_os_version is not None:
        windows_node_config = self.messages.WindowsNodeConfig()
        if options.windows_os_version == 'ltsc2022':
            windows_node_config.osVersion = self.messages.WindowsNodeConfig.OsVersionValueValuesEnum.OS_VERSION_LTSC2022
        else:
            windows_node_config.osVersion = self.messages.WindowsNodeConfig.OsVersionValueValuesEnum.OS_VERSION_LTSC2019
        update_request.windowsNodeConfig = windows_node_config
    elif options.resource_manager_tags is not None:
        tags = options.resource_manager_tags
        update_request.resourceManagerTags = self._ResourceManagerTags(tags)
    elif options.machine_type is not None or options.disk_type is not None or options.disk_size_gb is not None:
        update_request.machineType = options.machine_type
        update_request.diskType = options.disk_type
        update_request.diskSizeGb = options.disk_size_gb
    elif options.enable_queued_provisioning is not None:
        update_request.queuedProvisioning = self.messages.QueuedProvisioning()
        update_request.queuedProvisioning.enabled = options.enable_queued_provisioning
    return update_request