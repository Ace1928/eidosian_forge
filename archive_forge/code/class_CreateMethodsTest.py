import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
class CreateMethodsTest(utils.BaseTestCase):

    def setUp(self):
        super(CreateMethodsTest, self).setUp()
        self.client = mock.MagicMock()

    def test_create_single_node(self):
        params = {'driver': 'fake'}
        self.client.node.create.return_value = mock.Mock(uuid='uuid')
        self.assertEqual(('uuid', None), create_resources.create_single_node(self.client, **params))
        self.client.node.create.assert_called_once_with(driver='fake')

    def test_create_single_node_with_ports(self):
        params = {'driver': 'fake', 'ports': ['some ports here']}
        self.client.node.create.return_value = mock.Mock(uuid='uuid')
        self.assertEqual(('uuid', None), create_resources.create_single_node(self.client, **params))
        self.client.node.create.assert_called_once_with(driver='fake')

    def test_create_single_node_with_portgroups(self):
        params = {'driver': 'fake', 'portgroups': ['some portgroups']}
        self.client.node.create.return_value = mock.Mock(uuid='uuid')
        self.assertEqual(('uuid', None), create_resources.create_single_node(self.client, **params))
        self.client.node.create.assert_called_once_with(driver='fake')

    def test_create_single_node_raises_client_exception(self):
        params = {'driver': 'fake'}
        e = exc.ClientException('foo')
        self.client.node.create.side_effect = e
        res, err = create_resources.create_single_node(self.client, **params)
        self.assertIsNone(res)
        self.assertIsInstance(err, exc.ClientException)
        self.assertIn('Unable to create the node', str(err))
        self.client.node.create.assert_called_once_with(driver='fake')

    def test_create_single_node_raises_invalid_exception(self):
        params = {'driver': 'fake'}
        e = exc.InvalidAttribute('foo')
        self.client.node.create.side_effect = e
        res, err = create_resources.create_single_node(self.client, **params)
        self.assertIsNone(res)
        self.assertIsInstance(err, exc.InvalidAttribute)
        self.assertIn('Cannot create the node with attributes', str(err))
        self.client.node.create.assert_called_once_with(driver='fake')

    def test_create_single_port(self):
        params = {'address': 'fake-address', 'node_uuid': 'fake-node-uuid'}
        self.client.port.create.return_value = mock.Mock(uuid='fake-port-uuid')
        self.assertEqual(('fake-port-uuid', None), create_resources.create_single_port(self.client, **params))
        self.client.port.create.assert_called_once_with(**params)

    def test_create_single_portgroup(self):
        params = {'address': 'fake-address', 'node_uuid': 'fake-node-uuid'}
        self.client.portgroup.create.return_value = mock.Mock(uuid='fake-portgroup-uuid')
        self.assertEqual(('fake-portgroup-uuid', None), create_resources.create_single_portgroup(self.client, **params))
        self.client.portgroup.create.assert_called_once_with(**params)

    def test_create_single_portgroup_with_ports(self):
        params = {'ports': ['some ports'], 'node_uuid': 'fake-node-uuid'}
        self.client.portgroup.create.return_value = mock.Mock(uuid='fake-portgroup-uuid')
        self.assertEqual(('fake-portgroup-uuid', None), create_resources.create_single_portgroup(self.client, **params))
        self.client.portgroup.create.assert_called_once_with(node_uuid='fake-node-uuid')

    def test_create_single_chassis(self):
        self.client.chassis.create.return_value = mock.Mock(uuid='uuid')
        self.assertEqual(('uuid', None), create_resources.create_single_chassis(self.client))
        self.client.chassis.create.assert_called_once_with()

    def test_create_single_chassis_with_nodes(self):
        params = {'nodes': ['some nodes here']}
        self.client.chassis.create.return_value = mock.Mock(uuid='uuid')
        self.assertEqual(('uuid', None), create_resources.create_single_chassis(self.client, **params))
        self.client.chassis.create.assert_called_once_with()

    def test_create_ports(self):
        port = {'address': 'fake-address'}
        port_with_node_uuid = port.copy()
        port_with_node_uuid.update(node_uuid='fake-node-uuid')
        self.client.port.create.return_value = mock.Mock(uuid='uuid')
        self.assertEqual([], create_resources.create_ports(self.client, [port], 'fake-node-uuid'))
        self.client.port.create.assert_called_once_with(**port_with_node_uuid)

    def test_create_ports_two_node_uuids(self):
        port = {'address': 'fake-address', 'node_uuid': 'node-uuid-1'}
        errs = create_resources.create_ports(self.client, [port], 'node-uuid-2')
        self.assertIsInstance(errs[0], exc.ClientException)
        self.assertEqual(1, len(errs))
        self.assertFalse(self.client.port.create.called)

    def test_create_ports_two_portgroup_uuids(self):
        port = {'address': 'fake-address', 'node_uuid': 'node-uuid-1', 'portgroup_uuid': 'pg-uuid-1'}
        errs = create_resources.create_ports(self.client, [port], 'node-uuid-1', 'pg-uuid-2')
        self.assertEqual(1, len(errs))
        self.assertIsInstance(errs[0], exc.ClientException)
        self.assertIn('port group', str(errs[0]))
        self.assertFalse(self.client.port.create.called)

    @mock.patch.object(create_resources, 'create_portgroups', autospec=True)
    @mock.patch.object(create_resources, 'create_ports', autospec=True)
    def test_create_nodes(self, mock_create_ports, mock_create_portgroups):
        node = {'driver': 'fake', 'ports': ['list of ports'], 'portgroups': ['list of portgroups']}
        self.client.node.create.return_value = mock.Mock(uuid='uuid')
        self.assertEqual([], create_resources.create_nodes(self.client, [node]))
        self.client.node.create.assert_called_once_with(driver='fake')
        mock_create_ports.assert_called_once_with(self.client, ['list of ports'], node_uuid='uuid')
        mock_create_portgroups.assert_called_once_with(self.client, ['list of portgroups'], node_uuid='uuid')

    @mock.patch.object(create_resources, 'create_portgroups', autospec=True)
    @mock.patch.object(create_resources, 'create_ports', autospec=True)
    def test_create_nodes_exception(self, mock_create_ports, mock_create_portgroups):
        node = {'driver': 'fake', 'ports': ['list of ports'], 'portgroups': ['list of portgroups']}
        self.client.node.create.side_effect = exc.ClientException('bar')
        errs = create_resources.create_nodes(self.client, [node])
        self.assertIsInstance(errs[0], exc.ClientException)
        self.assertEqual(1, len(errs))
        self.client.node.create.assert_called_once_with(driver='fake')
        self.assertFalse(mock_create_ports.called)
        self.assertFalse(mock_create_portgroups.called)

    @mock.patch.object(create_resources, 'create_ports', autospec=True)
    def test_create_nodes_two_chassis_uuids(self, mock_create_ports):
        node = {'driver': 'fake', 'ports': ['list of ports'], 'chassis_uuid': 'chassis-uuid-1'}
        errs = create_resources.create_nodes(self.client, [node], chassis_uuid='chassis-uuid-2')
        self.assertFalse(self.client.node.create.called)
        self.assertFalse(mock_create_ports.called)
        self.assertEqual(1, len(errs))
        self.assertIsInstance(errs[0], exc.ClientException)

    @mock.patch.object(create_resources, 'create_portgroups', autospec=True)
    @mock.patch.object(create_resources, 'create_ports', autospec=True)
    def test_create_nodes_no_ports_portgroups(self, mock_create_ports, mock_create_portgroups):
        node = {'driver': 'fake'}
        self.client.node.create.return_value = mock.Mock(uuid='uuid')
        self.assertEqual([], create_resources.create_nodes(self.client, [node]))
        self.client.node.create.assert_called_once_with(driver='fake')
        self.assertFalse(mock_create_ports.called)
        self.assertFalse(mock_create_portgroups.called)

    @mock.patch.object(create_resources, 'create_nodes', autospec=True)
    def test_create_chassis(self, mock_create_nodes):
        chassis = {'description': 'fake', 'nodes': ['list of nodes']}
        self.client.chassis.create.return_value = mock.Mock(uuid='uuid')
        self.assertEqual([], create_resources.create_chassis(self.client, [chassis]))
        self.client.chassis.create.assert_called_once_with(description='fake')
        mock_create_nodes.assert_called_once_with(self.client, ['list of nodes'], chassis_uuid='uuid')

    @mock.patch.object(create_resources, 'create_nodes', autospec=True)
    def test_create_chassis_exception(self, mock_create_nodes):
        chassis = {'description': 'fake', 'nodes': ['list of nodes']}
        self.client.chassis.create.side_effect = exc.ClientException('bar')
        errs = create_resources.create_chassis(self.client, [chassis])
        self.client.chassis.create.assert_called_once_with(description='fake')
        self.assertFalse(mock_create_nodes.called)
        self.assertEqual(1, len(errs))
        self.assertIsInstance(errs[0], exc.ClientException)

    @mock.patch.object(create_resources, 'create_nodes', autospec=True)
    def test_create_chassis_no_nodes(self, mock_create_nodes):
        chassis = {'description': 'fake'}
        self.client.chassis.create.return_value = mock.Mock(uuid='uuid')
        self.assertEqual([], create_resources.create_chassis(self.client, [chassis]))
        self.client.chassis.create.assert_called_once_with(description='fake')
        self.assertFalse(mock_create_nodes.called)

    @mock.patch.object(create_resources, 'create_ports', autospec=True)
    def test_create_portgroups(self, mock_create_ports):
        portgroup = {'name': 'fake', 'ports': ['list of ports']}
        portgroup_posted = {'name': 'fake', 'node_uuid': 'fake-node-uuid'}
        self.client.portgroup.create.return_value = mock.Mock(uuid='uuid')
        self.assertEqual([], create_resources.create_portgroups(self.client, [portgroup], node_uuid='fake-node-uuid'))
        self.client.portgroup.create.assert_called_once_with(**portgroup_posted)
        mock_create_ports.assert_called_once_with(self.client, ['list of ports'], node_uuid='fake-node-uuid', portgroup_uuid='uuid')

    @mock.patch.object(create_resources, 'create_ports', autospec=True)
    def test_create_portgroups_exception(self, mock_create_ports):
        portgroup = {'name': 'fake', 'ports': ['list of ports']}
        portgroup_posted = {'name': 'fake', 'node_uuid': 'fake-node-uuid'}
        self.client.portgroup.create.side_effect = exc.ClientException('bar')
        errs = create_resources.create_portgroups(self.client, [portgroup], node_uuid='fake-node-uuid')
        self.client.portgroup.create.assert_called_once_with(**portgroup_posted)
        self.assertFalse(mock_create_ports.called)
        self.assertEqual(1, len(errs))
        self.assertIsInstance(errs[0], exc.ClientException)

    @mock.patch.object(create_resources, 'create_ports', autospec=True)
    def test_create_portgroups_two_node_uuids(self, mock_create_ports):
        portgroup = {'name': 'fake', 'node_uuid': 'fake-node-uuid-1', 'ports': ['list of ports']}
        self.client.portgroup.create.side_effect = exc.ClientException('bar')
        errs = create_resources.create_portgroups(self.client, [portgroup], node_uuid='fake-node-uuid-2')
        self.assertFalse(self.client.portgroup.create.called)
        self.assertFalse(mock_create_ports.called)
        self.assertEqual(1, len(errs))
        self.assertIsInstance(errs[0], exc.ClientException)

    @mock.patch.object(create_resources, 'create_ports', autospec=True)
    def test_create_portgroups_no_ports(self, mock_create_ports):
        portgroup = {'name': 'fake'}
        portgroup_posted = {'name': 'fake', 'node_uuid': 'fake-node-uuid'}
        self.client.portgroup.create.return_value = mock.Mock(uuid='uuid')
        self.assertEqual([], create_resources.create_portgroups(self.client, [portgroup], node_uuid='fake-node-uuid'))
        self.client.portgroup.create.assert_called_once_with(**portgroup_posted)
        self.assertFalse(mock_create_ports.called)