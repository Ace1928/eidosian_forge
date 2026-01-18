from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.filestore import util
from googlecloudsdk.core import properties
def FormatBackupAccessRequest(ref, args, req):
    """Python hook for yaml commands to supply backup access requests with the proper name."""
    del ref
    project = properties.VALUES.core.project.Get(required=True)
    location = args.region
    req.name = BACKUP_NAME_TEMPLATE.format(project, location, args.backup)
    return req