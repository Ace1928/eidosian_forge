import sys
import json
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.rcodezero import RcodeZeroDNSDriver
class RcodeZeroDNSMockHttp(MockHttp):
    fixtures = DNSFileFixtures('rcodezero')
    base_headers = {'content-type': 'application/json'}

    def _api_v1_zones(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('list_zones.json')
        elif method == 'POST':
            body = ''
        else:
            raise NotImplementedError('Unexpected method: %s' % method)
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _api_v1_zones_example_at(self, method, *args, **kwargs):
        if method == 'GET':
            body = self.fixtures.load('get_zone_details.json')
        elif method == 'DELETE':
            return (httplib.NO_CONTENT, '', self.base_headers, httplib.responses[httplib.NO_CONTENT])
        else:
            raise NotImplementedError('Unexpected method: %s' % method)
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _api_v1_zones_example_at__rrsets(self, method, *args, **kwargs):
        return self._api_v1_zones_example_at_rrsets(method, *args, **kwargs)

    def _api_v1_zones_example_at_rrsets(self, method, *args, **kwargs):
        if method == 'GET':
            body = self.fixtures.load('list_records.json')
        elif method == 'PATCH':
            body = ''
        else:
            raise NotImplementedError('Unexpected method: %s' % method)
        return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])

    def _api_v1_zones_EXISTS(self, method, url, body, headers):
        if method != 'POST':
            raise NotImplementedError('Unexpected method: %s' % method)
        payload = json.loads(body)
        domain = payload['domain']
        body = json.dumps({'error': "Domain '%s' already exists" % domain})
        return (httplib.UNPROCESSABLE_ENTITY, body, self.base_headers, 'Unprocessable Entity')

    def _api_v1_zones_example_com_MISSING(self, *args, **kwargs):
        return (httplib.NOT_FOUND, '{"status": "failed","message": "Zone not found"}', self.base_headers, 'Unprocessable Entity')

    def _api_v1_zones_example_at_MISSING(self, *args, **kwargs):
        return (httplib.NOT_FOUND, '{"status": "failed","message": "Zone not found"}', self.base_headers, 'Unprocessable Entity')