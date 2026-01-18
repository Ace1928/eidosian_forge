from unittest import mock
from openstack import exceptions
from heat.engine.clients.os import senlin as senlin_plugin
from heat.tests import common
from heat.tests import utils
class PolicyConstraintTest(common.HeatTestCase):

    @mock.patch('openstack.connection.Connection')
    def setUp(self, mock_connection):
        super(PolicyConstraintTest, self).setUp()
        self.senlin_client = mock.MagicMock()
        self.ctx = utils.dummy_context()
        self.mock_get_policy = mock.Mock()
        self.ctx.clients.client('senlin').get_policy = self.mock_get_policy
        self.constraint = senlin_plugin.PolicyConstraint()

    def test_validate_true(self):
        self.mock_get_policy.return_value = None
        self.assertTrue(self.constraint.validate('POLICY_ID', self.ctx))

    def test_validate_false(self):
        self.mock_get_policy.side_effect = exceptions.ResourceNotFound('POLICY_ID')
        self.assertFalse(self.constraint.validate('POLICY_ID', self.ctx))
        self.mock_get_policy.side_effect = exceptions.HttpException('POLICY_ID')
        self.assertFalse(self.constraint.validate('POLICY_ID', self.ctx))