from unittest import mock
from keystoneauth1 import exceptions as keystone_exceptions
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.tests import common
class KeystoneRegionConstraintTest(common.HeatTestCase):
    sample_uuid = '477e8273-60a7-4c41-b683-fdb0bc7cd151'

    def test_expected_exceptions(self):
        self.assertEqual((exception.EntityNotFound,), ks_constr.KeystoneRegionConstraint.expected_exceptions, 'KeystoneRegionConstraint expected exceptions error')

    def test_constraint(self):
        constraint = ks_constr.KeystoneRegionConstraint()
        client_mock = mock.MagicMock()
        client_plugin_mock = mock.MagicMock()
        client_plugin_mock.get_region_id.return_value = self.sample_uuid
        client_mock.client_plugin.return_value = client_plugin_mock
        self.assertIsNone(constraint.validate_with_client(client_mock, self.sample_uuid))
        self.assertRaises(exception.EntityNotFound, constraint.validate_with_client, client_mock, '')
        client_plugin_mock.get_region_id.assert_called_once_with(self.sample_uuid)