from unittest import mock
import uuid
from cinderclient import exceptions as cinder_exc
from keystoneauth1 import exceptions as ks_exceptions
from heat.common import exception
from heat.engine.clients.os import cinder
from heat.tests import common
from heat.tests import utils
class QoSSpecsConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(QoSSpecsConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.mock_get_qos_specs = mock.Mock()
        self.ctx.clients.client_plugin('cinder').get_qos_specs = self.mock_get_qos_specs
        self.constraint = cinder.QoSSpecsConstraint()

    def test_validation(self):
        self.mock_get_qos_specs.return_value = None
        self.assertTrue(self.constraint.validate('foo', self.ctx))

    def test_validation_error(self):
        self.mock_get_qos_specs.side_effect = cinder_exc.NotFound(404)
        self.assertFalse(self.constraint.validate('bar', self.ctx))