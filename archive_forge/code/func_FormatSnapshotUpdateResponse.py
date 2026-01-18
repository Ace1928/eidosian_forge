from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.command_lib.filestore import update_util
from googlecloudsdk.command_lib.filestore import util
from googlecloudsdk.core import properties
def FormatSnapshotUpdateResponse(response, args):
    """Python hook to generate the backup update response."""
    del response
    resource_ref = GetResourceRef(args)
    api_version = util.GetApiVersionFromArgs(args)
    client = filestore_client.FilestoreClient(api_version)
    return encoding.MessageToDict(client.GetInstanceSnapshot(resource_ref))