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
def _to_reserved_node(self, element):
    """
        Build an EC2ReservedNode object using the reserved instance properties.
        Information on these properties can be found at http://goo.gl/ulXCC7.
        """
    extra = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['reserved_node'])
    try:
        size = [size for size in self.list_sizes() if size.id == extra['instance_type']][0]
    except IndexError:
        size = None
    return EC2ReservedNode(id=findtext(element=element, xpath='reservedInstancesId', namespace=NAMESPACE), state=findattr(element=element, xpath='state', namespace=NAMESPACE), driver=self, size=size, extra=extra)