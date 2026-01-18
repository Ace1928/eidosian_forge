from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.backup.base import (
from libcloud.backup.types import BackupTargetType, BackupTargetJobStatusType
from libcloud.utils.iso8601 import parse_date
def _to_recovery_points(self, data, target):
    xpath = 'DescribeSnapshotsResponse/snapshotSet/item'
    return [self._to_recovery_point(el, target) for el in findall(element=data, xpath=xpath, namespace=NS)]