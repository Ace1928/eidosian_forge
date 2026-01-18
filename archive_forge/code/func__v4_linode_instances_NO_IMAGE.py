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
def _v4_linode_instances_NO_IMAGE(self, method, url, body, headers):
    body = self.fixtures.load('create_node_no_image.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])