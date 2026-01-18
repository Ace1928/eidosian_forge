from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.filestore import util
from googlecloudsdk.core import properties
def FormatBackupCreateRequest(ref, args, req):
    """Python hook for yaml commands to supply the backup create request with proper values."""
    del ref
    req.backupId = args.backup
    project = properties.VALUES.core.project.Get(required=True)
    location = args.region
    req.parent = PARENT_TEMPLATE.format(project, location)
    return req