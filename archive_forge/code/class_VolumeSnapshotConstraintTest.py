from unittest import mock
import uuid
from cinderclient import exceptions as cinder_exc
from keystoneauth1 import exceptions as ks_exceptions
from heat.common import exception
from heat.engine.clients.os import cinder
from heat.tests import common
from heat.tests import utils
class VolumeSnapshotConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(VolumeSnapshotConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.mock_get_snapshot = mock.Mock()
        self.ctx.clients.client_plugin('cinder').get_volume_snapshot = self.mock_get_snapshot
        self.constraint = cinder.VolumeSnapshotConstraint()

    def test_validation(self):
        self.mock_get_snapshot.return_value = 'snapshot'
        self.assertTrue(self.constraint.validate('foo', self.ctx))

    def test_validation_error(self):
        self.mock_get_snapshot.side_effect = exception.EntityNotFound(entity='VolumeSnapshot', name='bar')
        self.assertFalse(self.constraint.validate('bar', self.ctx))