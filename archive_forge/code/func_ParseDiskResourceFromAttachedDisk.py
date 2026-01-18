from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core.util import times
import six
def ParseDiskResourceFromAttachedDisk(resources, attached_disk):
    """Parses the source disk resource of an AttachedDisk.

  The source of an AttachedDisk is either a partial or fully specified URL
  referencing either a regional or zonal disk.

  Args:
    resources: resources.Registry, The resource registry
    attached_disk: AttachedDisk

  Returns:
    A disk resource.

  Raises:
    InvalidResourceException: If the attached disk source cannot be parsed as a
        regional or zonal disk.
  """
    try:
        disk = resources.Parse(attached_disk.source, collection='compute.regionDisks')
        if disk:
            return disk
    except (cloud_resources.WrongResourceCollectionException, cloud_resources.RequiredFieldOmittedException):
        pass
    try:
        disk = resources.Parse(attached_disk.source, collection='compute.disks')
        if disk:
            return disk
    except (cloud_resources.WrongResourceCollectionException, cloud_resources.RequiredFieldOmittedException):
        pass
    raise cloud_resources.InvalidResourceException('Unable to parse [{}]'.format(attached_disk.source))