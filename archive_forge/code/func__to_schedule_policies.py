from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def _to_schedule_policies(self, object):
    elements = object.findall(fixxpath('schedulePolicy', BACKUP_NS))
    return [self._to_schedule_policy(el) for el in elements]