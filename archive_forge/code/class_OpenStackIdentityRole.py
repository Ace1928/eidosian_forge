import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackIdentityRole:

    def __init__(self, id, name, description, enabled):
        self.id = id
        self.name = name
        self.description = description
        self.enabled = enabled

    def __repr__(self):
        return '<OpenStackIdentityRole id=%s, name=%s, description=%s, enabled=%s>' % (self.id, self.name, self.description, self.enabled)