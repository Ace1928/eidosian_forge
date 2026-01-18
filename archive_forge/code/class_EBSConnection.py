from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.backup.base import (
from libcloud.backup.types import BackupTargetType, BackupTargetJobStatusType
from libcloud.utils.iso8601 import parse_date
class EBSConnection(SignedAWSConnection):
    version = VERSION
    host = HOST
    responseCls = EBSResponse
    service_name = 'backup'