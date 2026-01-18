import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DURABLEDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.durabledns import (
def _services_dns_getRecord_php_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
    body = self.fixtures.load('get_record_ZONE_DOES_NOT_EXIST.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])