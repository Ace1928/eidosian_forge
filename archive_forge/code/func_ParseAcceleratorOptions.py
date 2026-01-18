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
def ParseAcceleratorOptions(self, options, node_config):
    """Parses accrelerator options for the nodes in the cluster."""
    if options.accelerators is not None:
        type_name = options.accelerators['type']
        count = int(options.accelerators.get('count', 1))
        accelerator_config = self.messages.AcceleratorConfig(acceleratorType=type_name, acceleratorCount=count)
        gpu_partition_size = options.accelerators.get('gpu-partition-size', '')
        if gpu_partition_size:
            accelerator_config.gpuPartitionSize = gpu_partition_size
        max_time_shared_clients_per_gpu = int(options.accelerators.get('max-time-shared-clients-per-gpu', 0))
        if max_time_shared_clients_per_gpu:
            accelerator_config.maxTimeSharedClientsPerGpu = max_time_shared_clients_per_gpu
        gpu_sharing_strategy = options.accelerators.get('gpu-sharing-strategy', None)
        max_shared_clients_per_gpu = options.accelerators.get('max-shared-clients-per-gpu', None)
        if max_shared_clients_per_gpu or gpu_sharing_strategy:
            if max_shared_clients_per_gpu is None:
                max_shared_clients_per_gpu = 2
            else:
                max_shared_clients_per_gpu = int(max_shared_clients_per_gpu)
            strategy_enum = self.messages.GPUSharingConfig.GpuSharingStrategyValueValuesEnum
            if gpu_sharing_strategy is None:
                gpu_sharing_strategy = strategy_enum.TIME_SHARING
            elif gpu_sharing_strategy == 'time-sharing':
                gpu_sharing_strategy = strategy_enum.TIME_SHARING
            elif gpu_sharing_strategy == 'mps':
                gpu_sharing_strategy = strategy_enum.MPS
            else:
                raise util.Error(GPU_SHARING_STRATEGY_ERROR_MSG)
            gpu_sharing_config = self.messages.GPUSharingConfig(maxSharedClientsPerGpu=max_shared_clients_per_gpu, gpuSharingStrategy=gpu_sharing_strategy)
            accelerator_config.gpuSharingConfig = gpu_sharing_config
        gpu_driver_version = options.accelerators.get('gpu-driver-version', None)
        if gpu_driver_version is not None:
            if gpu_driver_version.lower() == 'default':
                gpu_driver_version = self.messages.GPUDriverInstallationConfig.GpuDriverVersionValueValuesEnum.DEFAULT
            elif gpu_driver_version.lower() == 'latest':
                gpu_driver_version = self.messages.GPUDriverInstallationConfig.GpuDriverVersionValueValuesEnum.LATEST
            elif gpu_driver_version.lower() == 'disabled':
                gpu_driver_version = self.messages.GPUDriverInstallationConfig.GpuDriverVersionValueValuesEnum.INSTALLATION_DISABLED
            else:
                raise util.Error(GPU_DRIVER_VERSION_ERROR_MSG)
            gpu_driver_installation_config = self.messages.GPUDriverInstallationConfig(gpuDriverVersion=gpu_driver_version)
            accelerator_config.gpuDriverInstallationConfig = gpu_driver_installation_config
        node_config.accelerators = [accelerator_config]