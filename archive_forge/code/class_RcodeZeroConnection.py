import re
import json
import hashlib
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
from libcloud.common.exceptions import BaseHTTPError
class RcodeZeroConnection(ConnectionKey):
    responseCls = RcodeZeroResponse
    host = API_HOST

    def add_default_headers(self, headers):
        headers['Authorization'] = 'Bearer ' + self.key
        headers['Accept'] = 'application/json'
        return headers