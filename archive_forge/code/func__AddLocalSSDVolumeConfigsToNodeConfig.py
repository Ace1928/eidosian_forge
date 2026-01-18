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
def _AddLocalSSDVolumeConfigsToNodeConfig(self, node_config, options):
    """Add LocalSSDVolumeConfigs to nodeConfig."""
    if not options.local_ssd_volume_configs:
        return
    format_enum = self.messages.LocalSsdVolumeConfig.FormatValueValuesEnum
    local_ssd_volume_configs_list = []
    for config in options.local_ssd_volume_configs:
        count = int(config['count'])
        ssd_type = config['type'].lower()
        if config['format'].lower() == 'fs':
            ssd_format = format_enum.FS
        elif config['format'].lower() == 'block':
            ssd_format = format_enum.BLOCK
        else:
            raise util.Error(LOCAL_SSD_INCORRECT_FORMAT_ERROR_MSG.format(err_format=config['format']))
        local_ssd_volume_configs_list.append(self.messages.LocalSsdVolumeConfig(count=count, type=ssd_type, format=ssd_format))
    node_config.localSsdVolumeConfigs = local_ssd_volume_configs_list