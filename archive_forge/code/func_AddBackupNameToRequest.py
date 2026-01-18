from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.filestore import util
from googlecloudsdk.core import properties
def AddBackupNameToRequest(ref, args, req):
    """Python hook for yaml commands to process the source backup name."""
    del ref
    if args.source_backup is None or args.source_backup_region is None:
        return req
    project = properties.VALUES.core.project.Get(required=True)
    req.restoreInstanceRequest.sourceBackup = BACKUP_NAME_TEMPLATE.format(project, args.source_backup_region, args.source_backup)
    return req