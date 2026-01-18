from unittest import mock
from keystoneauth1 import exceptions as keystone_exceptions
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.tests import common
class KeystoneGroupConstraintTest(common.HeatTestCase):

    def test_expected_exceptions(self):
        self.assertEqual((exception.EntityNotFound,), ks_constr.KeystoneGroupConstraint.expected_exceptions, 'KeystoneGroupConstraint expected exceptions error')

    def test_constraint(self):
        constraint = ks_constr.KeystoneGroupConstraint()
        client_mock = mock.MagicMock()
        client_plugin_mock = mock.MagicMock()
        client_plugin_mock.get_group_id.return_value = None
        client_mock.client_plugin.return_value = client_plugin_mock
        self.assertIsNone(constraint.validate_with_client(client_mock, 'group_1'))
        self.assertRaises(exception.EntityNotFound, constraint.validate_with_client, client_mock, '')
        client_plugin_mock.get_group_id.assert_called_once_with('group_1')