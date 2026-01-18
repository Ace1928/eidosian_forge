from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
@staticmethod
def _client_to_client_id(backup_client):
    return dd_object_to_id(backup_client, DimensionDataBackupClient)