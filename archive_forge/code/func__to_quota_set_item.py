import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def _to_quota_set_item(self, obj):
    if obj:
        if isinstance(obj, OpenStack_2_QuotaSetItem):
            return obj
        elif isinstance(obj, dict):
            return OpenStack_2_QuotaSetItem(obj['in_use'], obj['limit'], obj['reserved'])
        elif isinstance(obj, int):
            return OpenStack_2_QuotaSetItem(0, obj, 0)
        else:
            return None
    else:
        return None