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
def _ModifySecurityGroupAttribute(self, method, url, body, headers):
    params = {'RegionId': self.test.region, 'SecurityGroupId': self.test.fake_security_group_id, 'SecurityGroupName': self.test.sg_name, 'Description': self.test.sg_description}
    self.assertUrlContainsQueryParams(url, params)
    resp_body = self.fixtures.load('modify_security_group_by_id.xml')
    return (httplib.OK, resp_body, {}, httplib.responses[httplib.OK])