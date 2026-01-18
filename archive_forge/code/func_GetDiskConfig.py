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
def GetDiskConfig(dataproc, boot_disk_type, boot_disk_size, num_local_ssds, local_ssd_interface):
    """Get dataproc cluster disk configuration.

  Args:
    dataproc: Dataproc object that contains client, messages, and resources
    boot_disk_type: Type of the boot disk
    boot_disk_size: Size of the boot disk
    num_local_ssds: Number of the Local SSDs
    local_ssd_interface: Interface used to attach local SSDs

  Returns:
    disk_config: Dataproc cluster disk configuration
  """
    return dataproc.messages.DiskConfig(bootDiskType=boot_disk_type, bootDiskSizeGb=boot_disk_size, numLocalSsds=num_local_ssds, localSsdInterface=local_ssd_interface)