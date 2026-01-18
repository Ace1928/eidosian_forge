import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.nsone import NsOneException
from libcloud.test.secrets import DNS_PARAMS_NSONE
from libcloud.dns.drivers.nsone import NsOneDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def _v1_zones_test_com_example_com_A_DELETE_RECORD_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
    body = self.fixtures.load('record_does_not_exist.json')
    return (404, body, {}, httplib.responses[httplib.OK])