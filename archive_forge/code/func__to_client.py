from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def _to_client(self, element, target):
    client_id = element.get('id')
    return DimensionDataBackupClient(id=client_id, type=self._to_client_type(element), status=element.get('status'), schedule_policy=findtext(element, 'schedulePolicyName', BACKUP_NS), storage_policy=findtext(element, 'storagePolicyName', BACKUP_NS), download_url=findtext(element, 'downloadUrl', BACKUP_NS), running_job=self._to_backup_job(element, target, client_id), alert=self._to_alert(element))