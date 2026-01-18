from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.backup.base import (
from libcloud.backup.types import BackupTargetType, BackupTargetJobStatusType
from libcloud.utils.iso8601 import parse_date
def _get_resource_tags(self, element):
    """
        Parse tags from the provided element and return a dictionary with
        key/value pairs.

        :rtype: ``dict``
        """
    tags = {}
    tag_set = findall(element=element, xpath='tagSet/item', namespace=NS)
    for tag in tag_set:
        key = findtext(element=tag, xpath='key', namespace=NS)
        value = findtext(element=tag, xpath='value', namespace=NS)
        tags[key] = value
    return tags