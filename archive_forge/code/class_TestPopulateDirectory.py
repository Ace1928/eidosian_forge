import json
import os
from unittest import mock
import testtools
from openstack.baremetal import configdrive
class TestPopulateDirectory(testtools.TestCase):

    def _check(self, metadata, user_data=None, network_data=None, vendor_data=None):
        with configdrive.populate_directory(metadata, user_data=user_data, network_data=network_data, vendor_data=vendor_data) as d:
            for version in ('2012-08-10', 'latest'):
                with open(os.path.join(d, 'openstack', version, 'meta_data.json')) as fp:
                    actual_metadata = json.load(fp)
                self.assertEqual(metadata, actual_metadata)
                network_data_file = os.path.join(d, 'openstack', version, 'network_data.json')
                user_data_file = os.path.join(d, 'openstack', version, 'user_data')
                vendor_data_file = os.path.join(d, 'openstack', version, 'vendor_data2.json')
                if network_data is None:
                    self.assertFalse(os.path.exists(network_data_file))
                else:
                    with open(network_data_file) as fp:
                        self.assertEqual(network_data, json.load(fp))
                if vendor_data is None:
                    self.assertFalse(os.path.exists(vendor_data_file))
                else:
                    with open(vendor_data_file) as fp:
                        self.assertEqual(vendor_data, json.load(fp))
                if user_data is None:
                    self.assertFalse(os.path.exists(user_data_file))
                else:
                    if isinstance(user_data, str):
                        user_data = user_data.encode()
                    with open(user_data_file, 'rb') as fp:
                        self.assertEqual(user_data, fp.read())
        self.assertFalse(os.path.exists(d))

    def test_without_user_data(self):
        self._check({'foo': 42})

    def test_with_user_data(self):
        self._check({'foo': 42}, b'I am user data')

    def test_with_user_data_as_string(self):
        self._check({'foo': 42}, u'I am user data')

    def test_with_network_data(self):
        self._check({'foo': 42}, network_data={'networks': {}})

    def test_with_vendor_data(self):
        self._check({'foo': 42}, vendor_data={'foo': 'bar'})