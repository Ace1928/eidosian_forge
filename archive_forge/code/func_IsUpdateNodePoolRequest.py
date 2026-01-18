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
def IsUpdateNodePoolRequest(self):
    return self.workload_metadata is not None or self.workload_metadata_from_node is not None or self.node_locations is not None or (self.max_surge_upgrade is not None) or (self.max_unavailable_upgrade is not None) or (self.system_config_from_file is not None) or (self.labels is not None) or (self.node_labels is not None) or (self.node_taints is not None) or (self.tags is not None) or (self.enable_private_nodes is not None) or (self.enable_gcfs is not None) or (self.gvnic is not None) or (self.enable_image_streaming is not None) or (self.enable_surge_upgrade is not None) or (self.enable_blue_green_upgrade is not None) or (self.node_pool_soak_duration is not None) or (self.standard_rollout_policy is not None) or (self.network_performance_config is not None) or (self.enable_confidential_nodes is not None) or (self.enable_fast_socket is not None) or (self.logging_variant is not None) or (self.windows_os_version is not None) or (self.accelerators is not None) or (self.resource_manager_tags is not None) or (self.containerd_config_from_file is not None) or (self.machine_type is not None) or (self.disk_type is not None) or (self.disk_size_gb is not None) or (self.enable_queued_provisioning is not None)