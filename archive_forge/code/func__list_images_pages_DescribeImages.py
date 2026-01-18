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
def _list_images_pages_DescribeImages(self, method, url, body, headers):
    if 'PageNumber=2' in url:
        resp_body = self.fixtures.load('pages_describe_images_page2.xml')
    else:
        resp_body = self.fixtures.load('pages_describe_images.xml')
    return (httplib.OK, resp_body, {}, httplib.responses[httplib.OK])