from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def _to_clients(self, object, target):
    elements = object.findall(fixxpath('backupClient', BACKUP_NS))
    return [self._to_client(el, target) for el in elements]