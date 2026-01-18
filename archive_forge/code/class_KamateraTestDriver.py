import sys
import json
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.compute import providers
from libcloud.utils.py3 import httplib
from libcloud.compute.base import NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.test.secrets import KAMATERA_PARAMS
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.kamatera import KamateraNodeDriver
class KamateraTestDriver(KamateraNodeDriver):

    def ex_wait_command(self, *args, **kwargs):
        kwargs['poll_interval_seconds'] = 0
        return KamateraNodeDriver.ex_wait_command(self, *args, **kwargs)