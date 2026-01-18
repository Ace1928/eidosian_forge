from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.backup.base import (
from libcloud.backup.types import BackupTargetType, BackupTargetJobStatusType
from libcloud.utils.iso8601 import parse_date
def _to_recovery_point(self, el, target):
    id = findtext(element=el, xpath='snapshotId', namespace=NS)
    date = parse_date(findtext(element=el, xpath='startTime', namespace=NS))
    tags = self._get_resource_tags(el)
    point = BackupTargetRecoveryPoint(id=id, date=date, target=target, driver=self.connection.driver, extra={'snapshot-id': id, 'tags': tags})
    return point