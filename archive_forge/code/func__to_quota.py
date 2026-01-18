import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
def _to_quota(self, elem):
    """
        To Quota
        """
    quota = {}
    for reference_quota_item in findall(element=elem, xpath='referenceQuotaSet/item', namespace=OUTSCALE_NAMESPACE):
        reference = findtext(element=reference_quota_item, xpath='reference', namespace=OUTSCALE_NAMESPACE)
        quota_set = []
        for quota_item in findall(element=reference_quota_item, xpath='quotaSet/item', namespace=OUTSCALE_NAMESPACE):
            ownerId = findtext(element=quota_item, xpath='ownerId', namespace=OUTSCALE_NAMESPACE)
            name = findtext(element=quota_item, xpath='name', namespace=OUTSCALE_NAMESPACE)
            displayName = findtext(element=quota_item, xpath='displayName', namespace=OUTSCALE_NAMESPACE)
            description = findtext(element=quota_item, xpath='description', namespace=OUTSCALE_NAMESPACE)
            groupName = findtext(element=quota_item, xpath='groupName', namespace=OUTSCALE_NAMESPACE)
            maxQuotaValue = findtext(element=quota_item, xpath='maxQuotaValue', namespace=OUTSCALE_NAMESPACE)
            usedQuotaValue = findtext(element=quota_item, xpath='usedQuotaValue', namespace=OUTSCALE_NAMESPACE)
            quota_set.append({'ownerId': ownerId, 'name': name, 'displayName': displayName, 'description': description, 'groupName': groupName, 'maxQuotaValue': maxQuotaValue, 'usedQuotaValue': usedQuotaValue})
        quota[reference] = quota_set
    return quota