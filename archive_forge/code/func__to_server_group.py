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
def _to_server_group(self, obj):
    policy = None
    if 'policy' in obj:
        policy = obj['policy']
    elif 'policies' in obj and obj['policies']:
        policy = obj['policies'][0]
    return OpenStack_2_ServerGroup(id=obj['id'], name=obj['name'], policy=policy, members=obj.get('members'), rules=obj.get('rules'), driver=self.connection.driver)