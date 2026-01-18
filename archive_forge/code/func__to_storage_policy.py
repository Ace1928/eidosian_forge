from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def _to_storage_policy(self, element):
    return DimensionDataBackupStoragePolicy(retention_period=int(element.get('retentionPeriodInDays')), name=element.get('name'), secondary_location=element.get('secondaryLocation'))