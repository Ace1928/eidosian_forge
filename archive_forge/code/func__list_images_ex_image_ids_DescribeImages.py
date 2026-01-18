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
def _list_images_ex_image_ids_DescribeImages(self, method, url, body, headers):
    params = {'ImageId': self.test.fake_image.id + ',not-existed'}
    self.assertUrlContainsQueryParams(url, params)
    return self._DescribeImages(method, url, body, headers)