from unittest import mock
from openstack import exceptions
from heat.engine.clients.os import senlin as senlin_plugin
from heat.tests import common
from heat.tests import utils
class PolicyTypeConstraintTest(common.HeatTestCase):

    @mock.patch('openstack.connection.Connection')
    def setUp(self, mock_connection):
        super(PolicyTypeConstraintTest, self).setUp()
        self.senlin_client = mock.MagicMock()
        self.ctx = utils.dummy_context()
        deletion_policy_type = mock.MagicMock()
        deletion_policy_type.name = 'senlin.policy.deletion-1.0'
        lb_policy_type = mock.MagicMock()
        lb_policy_type.name = 'senlin.policy.loadbalance-1.0'
        self.mock_policy_types = mock.Mock(return_value=[deletion_policy_type, lb_policy_type])
        self.ctx.clients.client('senlin').policy_types = self.mock_policy_types
        self.constraint = senlin_plugin.PolicyTypeConstraint()

    def test_validate_true(self):
        self.assertTrue(self.constraint.validate('senlin.policy.deletion-1.0', self.ctx))

    def test_validate_false(self):
        self.assertFalse(self.constraint.validate('Invalid_type', self.ctx))