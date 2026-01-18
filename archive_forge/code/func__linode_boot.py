import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume, NodeAuthSSHKey, NodeAuthPassword
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.linode import LinodeNodeDriver
def _linode_boot(self, method, url, body, headers):
    body = '{"ERRORARRAY":[],"ACTION":"linode.boot","DATA":{"JobID":1300}}'
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])