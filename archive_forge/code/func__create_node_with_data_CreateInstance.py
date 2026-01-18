import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.test.secrets import ECS_PARAMS
from libcloud.compute.types import NodeState, StorageVolumeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ecs import ECSDriver
def _create_node_with_data_CreateInstance(self, method, url, body, headers):
    params = {'SecurityGroupId': 'sg-28ou0f3xa', 'DataDisk.1.Size': '5', 'DataDisk.1.Category': 'cloud', 'DataDisk.1.DiskName': 'data1', 'DataDisk.1.Description': 'description', 'DataDisk.1.Device': '/dev/xvdb', 'DataDisk.1.DeleteWithInstance': 'true'}
    self.assertUrlContainsQueryParams(url, params)
    resp_body = self.fixtures.load('create_instance.xml')
    return (httplib.OK, resp_body, {}, httplib.responses[httplib.OK])