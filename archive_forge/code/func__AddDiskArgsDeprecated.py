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
def _AddDiskArgsDeprecated(parser, include_driver_pool_args=False):
    """Adds deprecated disk related args to the parser."""
    master_boot_disk_size = parser.add_mutually_exclusive_group()
    worker_boot_disk_size = parser.add_mutually_exclusive_group()
    master_boot_disk_size.add_argument('--master-boot-disk-size-gb', action=actions.DeprecationAction('--master-boot-disk-size-gb', warn='The `--master-boot-disk-size-gb` flag is deprecated. Use `--master-boot-disk-size` flag with "GB" after value.'), type=int, hidden=True, help='Use `--master-boot-disk-size` flag with "GB" after value.')
    worker_boot_disk_size.add_argument('--worker-boot-disk-size-gb', action=actions.DeprecationAction('--worker-boot-disk-size-gb', warn='The `--worker-boot-disk-size-gb` flag is deprecated. Use `--worker-boot-disk-size` flag with "GB" after value.'), type=int, hidden=True, help='Use `--worker-boot-disk-size` flag with "GB" after value.')
    boot_disk_size_detailed_help = "      The size of the boot disk. The value must be a\n      whole number followed by a size unit of ``KB'' for kilobyte, ``MB''\n      for megabyte, ``GB'' for gigabyte, or ``TB'' for terabyte. For example,\n      ``10GB'' will produce a 10 gigabyte disk. The minimum size a boot disk\n      can have is 10 GB. Disk size must be a multiple of 1 GB.\n      "
    master_boot_disk_size.add_argument('--master-boot-disk-size', type=arg_parsers.BinarySize(lower_bound='10GB'), help=boot_disk_size_detailed_help)
    worker_boot_disk_size.add_argument('--worker-boot-disk-size', type=arg_parsers.BinarySize(lower_bound='10GB'), help=boot_disk_size_detailed_help)
    secondary_worker_boot_disk_size = parser.add_argument_group(mutex=True)
    secondary_worker_boot_disk_size.add_argument('--preemptible-worker-boot-disk-size', type=arg_parsers.BinarySize(lower_bound='10GB'), help=boot_disk_size_detailed_help, hidden=True, action=actions.DeprecationAction('--preemptible-worker-boot-disk-size', warn='The `--preemptible-worker-boot-disk-size` flag is deprecated. Use the `--secondary-worker-boot-disk-size` flag instead.'))
    secondary_worker_boot_disk_size.add_argument('--secondary-worker-boot-disk-size', type=arg_parsers.BinarySize(lower_bound='10GB'), help=boot_disk_size_detailed_help)
    if include_driver_pool_args:
        parser.add_argument('--driver-pool-boot-disk-size', type=arg_parsers.BinarySize(lower_bound='10GB'), help=boot_disk_size_detailed_help)