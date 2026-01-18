from unittest import mock
from openstack import exceptions
from heat.engine.clients.os import senlin as senlin_plugin
from heat.tests import common
from heat.tests import utils
class ProfileTypeConstraintTest(common.HeatTestCase):

    @mock.patch('openstack.connection.Connection')
    def setUp(self, mock_connection):
        super(ProfileTypeConstraintTest, self).setUp()
        self.senlin_client = mock.MagicMock()
        self.ctx = utils.dummy_context()
        heat_profile_type = mock.MagicMock()
        heat_profile_type.name = 'os.heat.stack-1.0'
        nova_profile_type = mock.MagicMock()
        nova_profile_type.name = 'os.nova.server-1.0'
        self.mock_profile_types = mock.Mock(return_value=[heat_profile_type, nova_profile_type])
        self.ctx.clients.client('senlin').profile_types = self.mock_profile_types
        self.constraint = senlin_plugin.ProfileTypeConstraint()

    def test_validate_true(self):
        self.assertTrue(self.constraint.validate('os.heat.stack-1.0', self.ctx))

    def test_validate_false(self):
        self.assertFalse(self.constraint.validate('Invalid_type', self.ctx))