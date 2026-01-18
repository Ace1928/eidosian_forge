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
def _zones_1_records(self, method, url, body, headers):
    return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])