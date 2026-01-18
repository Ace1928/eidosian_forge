import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
class CloudSigmaAPI20DirectTestCase(CloudSigmaAPI20BaseTestCase, unittest.TestCase):
    driver_klass = CloudSigma_2_0_NodeDriver
    driver_args = ('foo', 'bar')
    driver_kwargs = {}