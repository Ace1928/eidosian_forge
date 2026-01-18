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
def _AddWorkloadMetadataToNodeConfig(self, node_config, options, messages):
    """Adds WorkLoadMetadata to NodeConfig."""
    if options.workload_metadata is not None:
        option = options.workload_metadata
        if option == 'GCE_METADATA':
            node_config.workloadMetadataConfig = messages.WorkloadMetadataConfig(mode=messages.WorkloadMetadataConfig.ModeValueValuesEnum.GCE_METADATA)
        elif option == 'GKE_METADATA':
            node_config.workloadMetadataConfig = messages.WorkloadMetadataConfig(mode=messages.WorkloadMetadataConfig.ModeValueValuesEnum.GKE_METADATA)
        else:
            raise util.Error(UNKNOWN_WORKLOAD_METADATA_ERROR_MSG.format(option=option))
    elif options.workload_metadata_from_node is not None:
        option = options.workload_metadata_from_node
        if option == 'GCE_METADATA':
            node_config.workloadMetadataConfig = messages.WorkloadMetadataConfig(mode=messages.WorkloadMetadataConfig.ModeValueValuesEnum.GCE_METADATA)
        elif option == 'GKE_METADATA':
            node_config.workloadMetadataConfig = messages.WorkloadMetadataConfig(mode=messages.WorkloadMetadataConfig.ModeValueValuesEnum.GKE_METADATA)
        elif option == 'SECURE':
            node_config.workloadMetadataConfig = messages.WorkloadMetadataConfig(nodeMetadata=messages.WorkloadMetadataConfig.NodeMetadataValueValuesEnum.SECURE)
        elif option == 'EXPOSED':
            node_config.workloadMetadataConfig = messages.WorkloadMetadataConfig(nodeMetadata=messages.WorkloadMetadataConfig.NodeMetadataValueValuesEnum.EXPOSE)
        elif option == 'GKE_METADATA_SERVER':
            node_config.workloadMetadataConfig = messages.WorkloadMetadataConfig(nodeMetadata=messages.WorkloadMetadataConfig.NodeMetadataValueValuesEnum.GKE_METADATA_SERVER)
        else:
            raise util.Error(UNKNOWN_WORKLOAD_METADATA_ERROR_MSG.format(option=option))