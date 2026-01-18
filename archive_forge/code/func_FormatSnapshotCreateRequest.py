from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.command_lib.filestore import update_util
from googlecloudsdk.command_lib.filestore import util
from googlecloudsdk.core import properties
def FormatSnapshotCreateRequest(ref, args, req):
    """Python hook for yaml commands to supply the snapshot create request with proper values."""
    del ref
    req.snapshotId = args.snapshot
    project = properties.VALUES.core.project.Get(required=True)
    location_id = args.instance_location if args.instance_region is None else args.instance_region
    req.parent = PARENT_TEMPLATE.format(project, location_id, args.instance)
    return req