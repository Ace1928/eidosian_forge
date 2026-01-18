from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.command_lib.filestore import update_util
from googlecloudsdk.command_lib.filestore import util
from googlecloudsdk.core import properties
def GetResourceRef(args):
    """Creates a Snapshot and returns its reference."""
    project = properties.VALUES.core.project.Get(required=True)
    api_version = util.GetApiVersionFromArgs(args)
    registry = filestore_client.GetFilestoreRegistry(api_version)
    location_id = args.instance_location if args.instance_region is None else args.instance_region
    ref = registry.Create('file.projects.locations.instances.snapshots', projectsId=project, locationsId=location_id, instancesId=args.instance, snapshotsId=args.snapshot)
    return ref