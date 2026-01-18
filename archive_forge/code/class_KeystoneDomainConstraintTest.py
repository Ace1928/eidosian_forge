from unittest import mock
from keystoneauth1 import exceptions as keystone_exceptions
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.tests import common
class KeystoneDomainConstraintTest(common.HeatTestCase):

    def test_expected_exceptions(self):
        self.assertEqual((exception.EntityNotFound,), ks_constr.KeystoneDomainConstraint.expected_exceptions, 'KeystoneDomainConstraint expected exceptions error')

    def test_constraint(self):
        constraint = ks_constr.KeystoneDomainConstraint()
        client_mock = mock.MagicMock()
        client_plugin_mock = mock.MagicMock()
        client_plugin_mock.get_domain_id.return_value = None
        client_mock.client_plugin.return_value = client_plugin_mock
        self.assertIsNone(constraint.validate_with_client(client_mock, 'domain_1'))
        self.assertRaises(exception.EntityNotFound, constraint.validate_with_client, client_mock, '')
        client_plugin_mock.get_domain_id.assert_called_once_with('domain_1')