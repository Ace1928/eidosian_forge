from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def _to_client_type(self, element):
    description = element.get('description')
    if description is None:
        description = findtext(element, 'description', BACKUP_NS)
    return DimensionDataBackupClientType(type=element.get('type'), description=description, is_file_system=bool(element.get('isFileSystem') == 'true'))