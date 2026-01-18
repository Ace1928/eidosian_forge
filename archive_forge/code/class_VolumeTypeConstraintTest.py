from unittest import mock
import uuid
from cinderclient import exceptions as cinder_exc
from keystoneauth1 import exceptions as ks_exceptions
from heat.common import exception
from heat.engine.clients.os import cinder
from heat.tests import common
from heat.tests import utils
class VolumeTypeConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(VolumeTypeConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.mock_get_volume_type = mock.Mock()
        self.ctx.clients.client_plugin('cinder').get_volume_type = self.mock_get_volume_type
        self.constraint = cinder.VolumeTypeConstraint()

    def test_validation(self):
        self.mock_get_volume_type.return_value = 'volume_type'
        self.assertTrue(self.constraint.validate('foo', self.ctx))

    def test_validation_error(self):
        self.mock_get_volume_type.side_effect = exception.EntityNotFound(entity='VolumeType', name='bar')
        self.assertFalse(self.constraint.validate('bar', self.ctx))