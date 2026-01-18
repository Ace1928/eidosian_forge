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
def ParseDiskType(resources, disk_type, project, location, scope, replica_zone_cnt=0):
    """Parses disk type reference based on location scope."""
    if scope == compute_scopes.ScopeEnum.ZONE:
        if replica_zone_cnt != 2:
            collection = 'compute.diskTypes'
            params = {'project': project, 'zone': location}
        else:
            collection = 'compute.regionDiskTypes'
            location = GetRegionFromZone(location)
            params = {'project': project, 'region': location}
    elif scope == compute_scopes.ScopeEnum.REGION:
        collection = 'compute.regionDiskTypes'
        params = {'project': project, 'region': location}
    disk_type_ref = resources.Parse(disk_type, collection=collection, params=params)
    return disk_type_ref