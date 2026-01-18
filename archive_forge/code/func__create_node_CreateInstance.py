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
def _create_node_CreateInstance(self, method, url, body, headers):
    params = {'SecurityGroupId': 'sg-28ou0f3xa', 'Description': 'description', 'InternetChargeType': 'PayByTraffic', 'InternetMaxBandwidthOut': '1', 'InternetMaxBandwidthIn': '200', 'HostName': 'hostname', 'Password': 'password', 'IoOptimized': 'optimized', 'SystemDisk.Category': 'cloud', 'SystemDisk.DiskName': 'root', 'SystemDisk.Description': 'sys', 'VSwitchId': 'vswitch-id1', 'PrivateIpAddress': '1.1.1.2', 'ClientToken': 'client_token'}
    self.assertUrlContainsQueryParams(url, params)
    resp_body = self.fixtures.load('create_instance.xml')
    return (httplib.OK, resp_body, {}, httplib.responses[httplib.OK])