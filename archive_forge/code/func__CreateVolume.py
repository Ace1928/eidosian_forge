import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
def _CreateVolume(self, method, url, body, headers):
    if 'KmsKeyId=' in url:
        assert 'Encrypted=1' in url, 'If a KmsKeyId is specified, the Encrypted flag must also be set.'
    if 'Encrypted=1' in url:
        body = self.fixtures.load('create_encrypted_volume.xml')
    else:
        body = self.fixtures.load('create_volume.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])