import sys
import json
from libcloud.test import MockHttp, unittest
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.test.secrets import DNS_PARAMS_CLOUDFLARE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.cloudflare import (
def _client_v4_zones_1234_dns_records_error_chain_error(self, method, url, body, headers):
    if method not in ['POST']:
        raise AssertionError('Unsupported method: %s' % method)
    body = self.fixtures.load('error_with_error_chain.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])