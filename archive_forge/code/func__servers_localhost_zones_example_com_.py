import sys
import json
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.powerdns import PowerDNSDriver
def _servers_localhost_zones_example_com_(self, method, *args, **kwargs):
    if method == 'GET':
        body = self.fixtures.load('list_records.json')
    elif method == 'PATCH':
        body = ''
    elif method == 'DELETE':
        return (httplib.NO_CONTENT, '', self.base_headers, httplib.responses[httplib.NO_CONTENT])
    else:
        raise NotImplementedError('Unexpected method: %s' % method)
    return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])