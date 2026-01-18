from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def _to_schedule_policy(self, element):
    return DimensionDataBackupSchedulePolicy(name=element.get('name'), description=element.get('description'))