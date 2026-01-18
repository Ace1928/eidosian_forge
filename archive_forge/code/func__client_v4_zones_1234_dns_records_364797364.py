import sys
import json
from libcloud.test import MockHttp, unittest
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.test.secrets import DNS_PARAMS_CLOUDFLARE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.cloudflare import (
def _client_v4_zones_1234_dns_records_364797364(self, method, url, body, headers):
    if method not in {'GET', 'PUT', 'DELETE'}:
        raise AssertionError('Unsupported method')
    body = self.fixtures.load('record_{}.json'.format(method))
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])