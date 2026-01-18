import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeImage, NodeState, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.common.linode import LinodeDisk, LinodeIPAddress, LinodeExceptionV4
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.linode import LinodeNodeDriver, LinodeNodeDriverV4
def _v4_linode_instances_22344420_ips(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('list_addresses_for_node.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    if method == 'POST':
        body = self.fixtures.load('allocate_private_address.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])