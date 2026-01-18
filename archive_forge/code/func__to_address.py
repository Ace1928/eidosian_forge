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
def _to_address(self, element, only_associated):
    instance_id = findtext(element=element, xpath='instanceId', namespace=NAMESPACE)
    public_ip = findtext(element=element, xpath='publicIp', namespace=NAMESPACE)
    domain = findtext(element=element, xpath='domain', namespace=NAMESPACE)
    extra = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['elastic_ip'])
    if only_associated and (not instance_id):
        return None
    return ElasticIP(public_ip, domain, instance_id, extra=extra)