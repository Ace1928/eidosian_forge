import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def _to_domains(self, data):
    result = []
    for item in data:
        domain = self._to_domain(data=item)
        result.append(domain)
    return result