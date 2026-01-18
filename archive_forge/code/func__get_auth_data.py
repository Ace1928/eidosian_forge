import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def _get_auth_data(self):
    data = {'auth': {'identity': {'methods': ['application_credential'], 'application_credential': {'id': self.user_id, 'secret': self.key}}}}
    return data