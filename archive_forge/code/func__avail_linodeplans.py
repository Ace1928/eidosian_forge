import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume, NodeAuthSSHKey, NodeAuthPassword
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.linode import LinodeNodeDriver
def _avail_linodeplans(self, method, url, body, headers):
    body = self.fixtures.load('_avail_linodeplans.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])