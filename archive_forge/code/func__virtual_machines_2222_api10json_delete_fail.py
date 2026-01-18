import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import VPSNET_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vpsnet import VPSNetNodeDriver
def _virtual_machines_2222_api10json_delete_fail(self, method, url, body, headers):
    return (httplib.FORBIDDEN, '', {}, httplib.responses[httplib.FORBIDDEN])