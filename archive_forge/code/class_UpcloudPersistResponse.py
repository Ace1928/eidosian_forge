import re
import sys
import json
import base64
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.compute import providers
from libcloud.utils.py3 import httplib, ensure_string
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.test.secrets import UPCLOUD_PARAMS
from libcloud.compute.types import Provider, NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.upcloud import UpcloudDriver, UpcloudResponse
class UpcloudPersistResponse(UpcloudResponse):

    def parse_body(self):
        import os
        path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, 'compute', 'fixtures', 'upcloud'))
        filename = 'api' + self.request.path_url.replace('/', '_').replace('.', '_') + '.json'
        filename = os.path.join(path, filename)
        if not os.path.exists(filename):
            with open(filename, 'w+') as f:
                f.write(self.body)
        return super().parse_body()