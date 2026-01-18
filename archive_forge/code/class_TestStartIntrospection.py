from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import node as _node
from openstack.baremetal_introspection.v1 import _proxy
from openstack.baremetal_introspection.v1 import introspection
from openstack.baremetal_introspection.v1 import introspection_rule
from openstack import exceptions
from openstack.tests.unit import base
from openstack.tests.unit import test_proxy_base
@mock.patch.object(introspection.Introspection, 'create', autospec=True)
class TestStartIntrospection(base.TestCase):

    def setUp(self):
        super(TestStartIntrospection, self).setUp()
        self.session = mock.Mock(spec=adapter.Adapter)
        self.proxy = _proxy.Proxy(self.session)

    def test_create_introspection(self, mock_create):
        self.proxy.start_introspection('abcd')
        mock_create.assert_called_once_with(mock.ANY, self.proxy)
        introspect = mock_create.call_args[0][0]
        self.assertEqual('abcd', introspect.id)

    def test_create_introspection_with_node(self, mock_create):
        self.proxy.start_introspection(_node.Node(id='abcd'))
        mock_create.assert_called_once_with(mock.ANY, self.proxy)
        introspect = mock_create.call_args[0][0]
        self.assertEqual('abcd', introspect.id)

    def test_create_introspection_manage_boot(self, mock_create):
        self.proxy.start_introspection('abcd', manage_boot=False)
        mock_create.assert_called_once_with(mock.ANY, self.proxy, manage_boot=False)
        introspect = mock_create.call_args[0][0]
        self.assertEqual('abcd', introspect.id)