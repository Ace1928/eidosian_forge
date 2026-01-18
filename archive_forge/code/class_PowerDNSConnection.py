import json
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
from libcloud.common.exceptions import BaseHTTPError
class PowerDNSConnection(ConnectionKey):
    responseCls = PowerDNSResponse

    def add_default_headers(self, headers):
        headers['X-API-Key'] = self.key
        return headers