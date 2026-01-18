from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def ValidateDiskSize(parameter_name, disk_size):
    """Validates that a disk size is a multiple of some number of GB.

  Args:
    parameter_name: parameter_name, the name of the parameter, formatted as it
      would be in help text (e.g., `--disk-size` or 'DISK_SIZE')
    disk_size: int, the disk size in bytes, or None for default value

  Raises:
    exceptions.InvalidArgumentException: the disk size was invalid
  """
    gb_mask = (1 << 30) - 1
    if disk_size and disk_size & gb_mask:
        raise exceptions.InvalidArgumentException(parameter_name, 'Must be an integer quantity of GB.')