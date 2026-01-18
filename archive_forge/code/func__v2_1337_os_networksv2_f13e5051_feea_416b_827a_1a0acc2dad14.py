import sys
import unittest
from libcloud.pricing import clear_pricing_data
from libcloud.utils.py3 import httplib, method_type
from libcloud.test.secrets import RACKSPACE_PARAMS, RACKSPACE_NOVA_PARAMS
from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver
from libcloud.compute.drivers.rackspace import RackspaceNodeDriver, RackspaceFirstGenNodeDriver
from libcloud.test.compute.test_openstack import (
def _v2_1337_os_networksv2_f13e5051_feea_416b_827a_1a0acc2dad14(self, method, url, body, headers):
    if method == 'DELETE':
        body = ''
        return (httplib.ACCEPTED, body, self.json_content_headers, httplib.responses[httplib.OK])
    raise NotImplementedError()