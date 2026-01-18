import sys
import json
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.powerdns import PowerDNSDriver
def _servers_localhost_zones_example_com__MISSING(self, *args, **kwargs):
    return (httplib.UNPROCESSABLE_ENTITY, 'Could not find domain', self.base_headers, 'Unprocessable Entity')