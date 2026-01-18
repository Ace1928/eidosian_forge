import sys
import json
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.rcodezero import RcodeZeroDNSDriver
def _api_v1_zones_example_at(self, method, *args, **kwargs):
    if method == 'GET':
        body = self.fixtures.load('get_zone_details.json')
    elif method == 'DELETE':
        return (httplib.NO_CONTENT, '', self.base_headers, httplib.responses[httplib.NO_CONTENT])
    else:
        raise NotImplementedError('Unexpected method: %s' % method)
    return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])