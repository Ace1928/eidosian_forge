import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts
from libcloud.compute.base import Node
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import CloudSigmaNodeDriver, CloudSigmaZrhNodeDriver
def _servers_62fe7cde_4fb9_4c63_bd8c_e757930066a0_set(self, method, url, body, headers):
    body = self.fixtures.load('servers_set.txt')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])