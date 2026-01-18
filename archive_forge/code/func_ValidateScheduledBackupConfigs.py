from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateScheduledBackupConfigs(unused_ref, args, req):
    """Validates that the cron_schedule and backup_location are set when the scheduled backup is enabled.

  Args:
    unused_ref: A resource ref to the parsed metastore service resource.
    args: The parsed args namespace from CLI.
    req: A request with `service` field.

  Returns:
    A request with service scheduled backups configurations required.
  Raises:
    BadArgumentException: when cron_schedule and backup_location are not set
    when the scheduled backup is enabled.
  """
    args_set = set(args.GetSpecifiedArgNames())
    if req.service.scheduledBackup.enabled and '--scheduled-backup-cron' not in args_set:
        raise exceptions.BadArgumentException('--scheduled-backup-cron', '--scheduled-backup-cron must be set when the scheduled backup is enabled.')
    if req.service.scheduledBackup.enabled and '--scheduled-backup-location' not in args_set:
        raise exceptions.BadArgumentException('--scheduled-backup-location', '--scheduled-backup-location must be set when the scheduled backup is enabled.')
    return req