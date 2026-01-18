import json
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
from libcloud.common.exceptions import BaseHTTPError
def _pdns_version(self):
    if self.api_root == '':
        return 3
    elif self.api_root == '/api/v1':
        return 4
    raise ValueError('PowerDNS version has not been declared')