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
class TestProxyList(base.TestCase):

    def setUp(self):
        super(TestProxyList, self).setUp()
        self.session = mock.Mock()
        self.args = {'a': 'A', 'b': 'B', 'c': 'C'}
        self.fake_response = [resource.Resource()]
        self.sot = proxy.Proxy(self.session)
        self.sot._connection = self.cloud
        ListableResource.list = mock.Mock()
        ListableResource.list.return_value = self.fake_response

    def _test_list(self, paginated, base_path=None):
        rv = self.sot._list(ListableResource, paginated=paginated, base_path=base_path, **self.args)
        self.assertEqual(self.fake_response, rv)
        ListableResource.list.assert_called_once_with(self.sot, paginated=paginated, base_path=base_path, **self.args)

    def test_list_paginated(self):
        self._test_list(True)

    def test_list_non_paginated(self):
        self._test_list(False)

    def test_list_override_base_path(self):
        self._test_list(False, base_path='dummy')

    def test_list_filters_jmespath(self):
        fake_response = [FilterableResource(a='a1', b='b1', c='c'), FilterableResource(a='a2', b='b2', c='c'), FilterableResource(a='a3', b='b3', c='c')]
        FilterableResource.list = mock.Mock()
        FilterableResource.list.return_value = fake_response
        rv = self.sot._list(FilterableResource, paginated=False, base_path=None, jmespath_filters="[?c=='c']")
        self.assertEqual(3, len(rv))
        rv = self.sot._list(FilterableResource, paginated=False, base_path=None, jmespath_filters="[?d=='c']")
        self.assertEqual(0, len(rv))