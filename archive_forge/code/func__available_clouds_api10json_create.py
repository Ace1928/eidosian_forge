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
def _available_clouds_api10json_create(self, method, url, body, headers):
    body = '\n        [{"cloud":{"system_templates":[{"id":9,"label":"Ubuntu 8.04 x64"}],"id":2,"label":"USA VPS Cloud"}}]\n        '
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])