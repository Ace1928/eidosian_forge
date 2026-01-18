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
def _to_subnet(self, element, name=None):
    subnet_id = findtext(element=element, xpath='subnetId', namespace=NAMESPACE)
    tags = self._get_resource_tags(element)
    name = name if name else tags.get('Name', subnet_id)
    state = findtext(element=element, xpath='state', namespace=NAMESPACE)
    extra = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['subnet'])
    extra['tags'] = tags
    return EC2NetworkSubnet(subnet_id, name, state, extra=extra)