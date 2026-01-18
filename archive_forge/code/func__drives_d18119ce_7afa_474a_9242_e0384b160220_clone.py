import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts
from libcloud.compute.base import Node
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import CloudSigmaNodeDriver, CloudSigmaZrhNodeDriver
def _drives_d18119ce_7afa_474a_9242_e0384b160220_clone(self, method, url, body, headers):
    body = self.fixtures.load('drives_clone.txt')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])