from unittest import mock
from ironicclient import exceptions as ic_exc
from heat.engine.clients.os import ironic as ic
from heat.tests import common
from heat.tests import utils
class PortGroupConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(PortGroupConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.mock_port_group_get = mock.Mock()
        self.ctx.clients.client_plugin('ironic').client().portgroup.get = self.mock_port_group_get
        self.constraint = ic.PortGroupConstraint()

    def test_validate(self):
        self.mock_port_group_get.return_value = fake_resource(id='my_port_group')
        self.assertTrue(self.constraint.validate('my_port_group', self.ctx))

    def test_validate_fail(self):
        self.mock_port_group_get.side_effect = ic_exc.NotFound()
        self.assertFalse(self.constraint.validate('bad_port_group', self.ctx))