import concurrent.futures
import hashlib
import logging
import sys
from unittest import mock
import fixtures
import os_service_types
import testtools
import openstack
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
class TestSupportsMicroversion(base.TestCase):

    def setUp(self):
        super(TestSupportsMicroversion, self).setUp()
        self.adapter = mock.Mock(spec=['get_endpoint_data'])
        self.endpoint_data = mock.Mock(spec=['min_microversion', 'max_microversion'], min_microversion='1.1', max_microversion='1.99')
        self.adapter.get_endpoint_data.return_value = self.endpoint_data

    def test_requested_supported_no_default(self):
        self.adapter.default_microversion = None
        self.assertTrue(utils.supports_microversion(self.adapter, '1.2'))

    def test_requested_not_supported_no_default(self):
        self.adapter.default_microversion = None
        self.assertFalse(utils.supports_microversion(self.adapter, '2.2'))

    def test_requested_not_supported_no_default_exception(self):
        self.adapter.default_microversion = None
        self.assertRaises(exceptions.SDKException, utils.supports_microversion, self.adapter, '2.2', True)

    def test_requested_supported_higher_default(self):
        self.adapter.default_microversion = '1.8'
        self.assertTrue(utils.supports_microversion(self.adapter, '1.6'))

    def test_requested_supported_equal_default(self):
        self.adapter.default_microversion = '1.8'
        self.assertTrue(utils.supports_microversion(self.adapter, '1.8'))

    def test_requested_supported_lower_default(self):
        self.adapter.default_microversion = '1.2'
        self.assertFalse(utils.supports_microversion(self.adapter, '1.8'))

    def test_requested_supported_lower_default_exception(self):
        self.adapter.default_microversion = '1.2'
        self.assertRaises(exceptions.SDKException, utils.supports_microversion, self.adapter, '1.8', True)

    @mock.patch('openstack.utils.supports_microversion')
    def test_require_microversion(self, sm_mock):
        utils.require_microversion(self.adapter, '1.2')
        sm_mock.assert_called_with(self.adapter, '1.2', raise_exception=True)