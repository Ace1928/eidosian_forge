import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.rimuhosting import RimuHostingNodeDriver
def _r_orders(self, method, url, body, headers):
    body = self.fixtures.load('r_orders.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])