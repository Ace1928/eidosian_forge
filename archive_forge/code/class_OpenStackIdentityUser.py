import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackIdentityUser:

    def __init__(self, id, domain_id, name, email, description, enabled):
        self.id = id
        self.domain_id = domain_id
        self.name = name
        self.email = email
        self.description = description
        self.enabled = enabled

    def __repr__(self):
        return '<OpenStackIdentityUser id=%s, domain_id=%s, name=%s, email=%s, enabled=%s>' % (self.id, self.domain_id, self.name, self.email, self.enabled)