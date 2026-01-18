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
def _list_nodes_ex_node_ids_DescribeInstances(self, method, url, body, headers):
    params = {'InstanceIds': '["i-28n7dkvov", "not-existed-id"]'}
    self.assertUrlContainsQueryParams(url, params)
    return self._DescribeInstances(method, url, body, headers)