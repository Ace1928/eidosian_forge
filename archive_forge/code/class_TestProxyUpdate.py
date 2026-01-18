import copy
import queue
from unittest import mock
from keystoneauth1 import session
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
class TestProxyUpdate(base.TestCase):

    def setUp(self):
        super(TestProxyUpdate, self).setUp()
        self.session = mock.Mock()
        self.fake_id = 1
        self.fake_result = 'fake_result'
        self.res = mock.Mock(spec=UpdateableResource)
        self.res.commit = mock.Mock(return_value=self.fake_result)
        self.sot = proxy.Proxy(self.session)
        self.sot._connection = self.cloud
        self.attrs = {'x': 1, 'y': 2, 'z': 3}
        UpdateableResource.new = mock.Mock(return_value=self.res)

    def test_update_resource(self):
        rv = self.sot._update(UpdateableResource, self.res, **self.attrs)
        self.assertEqual(rv, self.fake_result)
        self.res._update.assert_called_once_with(**self.attrs)
        self.res.commit.assert_called_once_with(self.sot, base_path=None)

    def test_update_resource_override_base_path(self):
        base_path = 'dummy'
        rv = self.sot._update(UpdateableResource, self.res, base_path=base_path, **self.attrs)
        self.assertEqual(rv, self.fake_result)
        self.res._update.assert_called_once_with(**self.attrs)
        self.res.commit.assert_called_once_with(self.sot, base_path=base_path)

    def test_update_id(self):
        rv = self.sot._update(UpdateableResource, self.fake_id, **self.attrs)
        self.assertEqual(rv, self.fake_result)
        self.res.commit.assert_called_once_with(self.sot, base_path=None)