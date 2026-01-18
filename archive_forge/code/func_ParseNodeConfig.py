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
def ParseNodeConfig(self, options):
    """Creates node config based on node config options."""
    node_config = self.messages.NodeConfig()
    if options.node_machine_type:
        node_config.machineType = options.node_machine_type
    if options.node_disk_size_gb:
        node_config.diskSizeGb = options.node_disk_size_gb
    if options.disk_type:
        node_config.diskType = options.disk_type
    if options.node_source_image:
        raise util.Error('cannot specify node source image in container v1 api')
    NodeIdentityOptionsToNodeConfig(options, node_config)
    if options.local_ssd_count:
        node_config.localSsdCount = options.local_ssd_count
    self._AddLocalSSDVolumeConfigsToNodeConfig(node_config, options)
    self._AddEphemeralStorageToNodeConfig(node_config, options)
    self._AddEphemeralStorageLocalSsdToNodeConfig(node_config, options)
    self._AddLocalNvmeSsdBlockToNodeConfig(node_config, options)
    self._AddEnableConfidentialStorageToNodeConfig(node_config, options)
    self._AddStoragePoolsToNodeConfig(node_config, options)
    if options.tags:
        node_config.tags = options.tags
    else:
        node_config.tags = []
    if options.image_type:
        node_config.imageType = options.image_type
    self.ParseCustomNodeConfig(options, node_config)
    _AddNodeLabelsToNodeConfig(node_config, options)
    _AddLabelsToNodeConfig(node_config, options)
    _AddMetadataToNodeConfig(node_config, options)
    self._AddNodeTaintsToNodeConfig(node_config, options)
    if options.resource_manager_tags is not None:
        tags = options.resource_manager_tags
        node_config.resourceManagerTags = self._ResourceManagerTags(tags)
    if options.preemptible:
        node_config.preemptible = options.preemptible
    if options.spot:
        node_config.spot = options.spot
    self.ParseAcceleratorOptions(options, node_config)
    if options.min_cpu_platform is not None:
        node_config.minCpuPlatform = options.min_cpu_platform
    self._AddWorkloadMetadataToNodeConfig(node_config, options, self.messages)
    _AddLinuxNodeConfigToNodeConfig(node_config, options, self.messages)
    _AddShieldedInstanceConfigToNodeConfig(node_config, options, self.messages)
    _AddReservationAffinityToNodeConfig(node_config, options, self.messages)
    if options.system_config_from_file is not None:
        util.LoadSystemConfigFromYAML(node_config, options.system_config_from_file, options.enable_insecure_kubelet_readonly_port, self.messages)
        if node_config.kubeletConfig.insecureKubeletReadonlyPortEnabled is not None and options.enable_insecure_kubelet_readonly_port is None:
            options.enable_insecure_kubelet_readonly_port = node_config.kubeletConfig.insecureKubeletReadonlyPortEnabled
    self.ParseAdvancedMachineFeatures(options, node_config)
    if options.gvnic is not None:
        gvnic = self.messages.VirtualNIC(enabled=options.gvnic)
        node_config.gvnic = gvnic
    return node_config