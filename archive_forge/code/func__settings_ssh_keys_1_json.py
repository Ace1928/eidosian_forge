import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.secrets import ONAPP_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.onapp import OnAppNodeDriver
def _settings_ssh_keys_1_json(self, method, url, body, headers):
    return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])