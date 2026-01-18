import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def _to_role(self, data):
    role = OpenStackIdentityRole(id=data['id'], name=data['name'], description=data.get('description', None), enabled=data.get('enabled', True))
    return role