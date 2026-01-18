import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import NodeImage
from libcloud.test.secrets import DIGITALOCEAN_v1_PARAMS, DIGITALOCEAN_v2_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.digitalocean import DigitalOcean_v1_Error
from libcloud.compute.drivers.digitalocean import DigitalOceanNodeDriver
def _v2_volumes_62766883_2c28_11e6_b8e6_000f53306ae1_snapshots_CREATE(self, method, url, body, headers):
    body = self.fixtures.load('create_volume_snapshot.json')
    return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])