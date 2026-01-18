from unittest import mock
from openstack.baremetal.v1 import _proxy
from openstack.baremetal.v1 import allocation
from openstack.baremetal.v1 import chassis
from openstack.baremetal.v1 import driver
from openstack.baremetal.v1 import node
from openstack.baremetal.v1 import port
from openstack.baremetal.v1 import port_group
from openstack.baremetal.v1 import volume_connector
from openstack.baremetal.v1 import volume_target
from openstack import exceptions
from openstack.tests.unit import base
from openstack.tests.unit import test_proxy_base
@mock.patch('time.sleep', lambda _sec: None)
@mock.patch.object(_proxy.Proxy, 'get_node', autospec=True)
class TestWaitForNodesProvisionState(base.TestCase):

    def setUp(self):
        super(TestWaitForNodesProvisionState, self).setUp()
        self.session = mock.Mock()
        self.proxy = _proxy.Proxy(self.session)

    def test_success(self, mock_get):
        nodes = [mock.Mock(spec=node.Node, id=str(i)) for i in range(3)]
        for i, n in enumerate(nodes):
            n._check_state_reached.return_value = not i % 2
            mock_get.side_effect = nodes
        result = self.proxy.wait_for_nodes_provision_state(['abcd', node.Node(id='1234')], 'fake state')
        self.assertEqual([nodes[0], nodes[2]], result)
        for n in nodes:
            n._check_state_reached.assert_called_once_with(self.proxy, 'fake state', True)

    def test_success_no_fail(self, mock_get):
        nodes = [mock.Mock(spec=node.Node, id=str(i)) for i in range(3)]
        for i, n in enumerate(nodes):
            n._check_state_reached.return_value = not i % 2
            mock_get.side_effect = nodes
        result = self.proxy.wait_for_nodes_provision_state(['abcd', node.Node(id='1234')], 'fake state', fail=False)
        self.assertEqual([nodes[0], nodes[2]], result.success)
        self.assertEqual([], result.failure)
        self.assertEqual([], result.timeout)
        for n in nodes:
            n._check_state_reached.assert_called_once_with(self.proxy, 'fake state', True)

    def test_timeout(self, mock_get):
        mock_get.return_value._check_state_reached.return_value = False
        mock_get.return_value.id = '1234'
        self.assertRaises(exceptions.ResourceTimeout, self.proxy.wait_for_nodes_provision_state, ['abcd', node.Node(id='1234')], 'fake state', timeout=0.001)
        mock_get.return_value._check_state_reached.assert_called_with(self.proxy, 'fake state', True)

    def test_timeout_no_fail(self, mock_get):
        mock_get.return_value._check_state_reached.return_value = False
        mock_get.return_value.id = '1234'
        result = self.proxy.wait_for_nodes_provision_state(['abcd'], 'fake state', timeout=0.001, fail=False)
        mock_get.return_value._check_state_reached.assert_called_with(self.proxy, 'fake state', True)
        self.assertEqual([], result.success)
        self.assertEqual([mock_get.return_value], result.timeout)
        self.assertEqual([], result.failure)

    def test_timeout_and_failures_not_fail(self, mock_get):

        def _fake_get(_self, node):
            result = mock.Mock()
            result.id = getattr(node, 'id', node)
            if result.id == '1':
                result._check_state_reached.return_value = True
            elif result.id == '2':
                result._check_state_reached.side_effect = exceptions.ResourceFailure('boom')
            else:
                result._check_state_reached.return_value = False
            return result
        mock_get.side_effect = _fake_get
        result = self.proxy.wait_for_nodes_provision_state(['1', '2', '3'], 'fake state', timeout=0.001, fail=False)
        self.assertEqual(['1'], [x.id for x in result.success])
        self.assertEqual(['3'], [x.id for x in result.timeout])
        self.assertEqual(['2'], [x.id for x in result.failure])