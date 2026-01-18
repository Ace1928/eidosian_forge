import sys
import unittest
from libcloud.pricing import clear_pricing_data
from libcloud.utils.py3 import httplib, method_type
from libcloud.test.secrets import RACKSPACE_PARAMS, RACKSPACE_NOVA_PARAMS
from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver
from libcloud.compute.drivers.rackspace import RackspaceNodeDriver, RackspaceFirstGenNodeDriver
from libcloud.test.compute.test_openstack import (
class RackspaceusFirstGenUkTests(OpenStack_1_0_Tests):
    should_list_locations = True
    should_have_pricing = True
    driver_klass = RackspaceFirstGenNodeDriver
    driver_type = RackspaceFirstGenNodeDriver
    driver_args = RACKSPACE_PARAMS
    driver_kwargs = {'region': 'uk'}

    def test_list_sizes_pricing(self):
        sizes = self.driver.list_sizes()
        for size in sizes:
            self.assertTrue(size.price > 0)