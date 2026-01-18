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
def _copy_image_same_region_CopyImage(self, method, url, body, headers):
    params = {'RegionId': self.test.region, 'ImageId': self.test.fake_image.id, 'DestinationRegionId': self.test.region}
    self.assertUrlContainsQueryParams(url, params)
    resp_body = self.fixtures.load('copy_image.xml')
    return (httplib.OK, resp_body, {}, httplib.responses[httplib.OK])