import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackIdentityProject:

    def __init__(self, id, name, description, enabled, domain_id=None):
        self.id = id
        self.name = name
        self.description = description
        self.enabled = enabled
        self.domain_id = domain_id

    def __repr__(self):
        return '<OpenStackIdentityProject id=%s, domain_id=%s, name=%s, enabled=%s>' % (self.id, self.domain_id, self.name, self.enabled)