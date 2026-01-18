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
def _create_ex_blockdevicemappings_RunInstances(self, method, url, body, headers):
    expected_params = {'BlockDeviceMapping.1.DeviceName': '/dev/sda1', 'BlockDeviceMapping.1.Ebs.VolumeSize': '10', 'BlockDeviceMapping.2.DeviceName': '/dev/sdb', 'BlockDeviceMapping.2.VirtualName': 'ephemeral0', 'BlockDeviceMapping.3.DeviceName': '/dev/sdc', 'BlockDeviceMapping.3.VirtualName': 'ephemeral1'}
    self.assertUrlContainsQueryParams(url, expected_params)
    body = self.fixtures.load('run_instances.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])