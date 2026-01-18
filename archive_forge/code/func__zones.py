import sys
import json
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.dns.base import Zone
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError
from libcloud.test.secrets import DNS_PARAMS_AURORADNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.auroradns import AuroraDNSDriver, AuroraDNSHealthCheckType
def _zones(self, method, url, body, headers):
    if method == 'POST':
        body_json = json.loads(body)
        if body_json['name'] == 'exists.example.com':
            return (httplib.CONFLICT, body, {}, httplib.responses[httplib.CONFLICT])
        body = self.fixtures.load('zone_example_com.json')
    else:
        body = self.fixtures.load('zone_list.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])