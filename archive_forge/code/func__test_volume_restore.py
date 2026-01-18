import collections
import copy
import json
from unittest import mock
from cinderclient import exceptions as cinder_exp
from novaclient import exceptions as nova_exp
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.resources.openstack.cinder import volume as c_vol
from heat.engine.resources import scheduler_hints as sh
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.objects import resource_data as resource_data_object
from heat.tests.openstack.cinder import test_volume_utils as vt_base
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _test_volume_restore(self, stack_name, final_status='available', stack_final_status=('RESTORE', 'COMPLETE')):
    self.cinder_fc.volumes.create.return_value = vt_base.FakeVolume('creating')
    fv = vt_base.FakeVolume('available')
    fv_restoring = vt_base.FakeVolume('restoring-backup', id=fv.id, attachments=[])
    fv_final = vt_base.FakeVolume(final_status, id=fv.id)
    self.cinder_fc.volumes.get.side_effect = [fv, fv_restoring, fv_final]
    self.stub_VolumeBackupConstraint_validate()
    fb = vt_base.FakeBackup('creating')
    self.patchobject(self.cinder_fc, 'backups')
    self.cinder_fc.backups.create.return_value = fb
    self.cinder_fc.backups.get.return_value = vt_base.FakeBackup('available')
    fvbr = vt_base.FakeBackupRestore('vol-123')
    self.patchobject(self.cinder_fc.restores, 'restore')
    self.cinder_fc.restores.restore.return_value = fvbr
    t = template_format.parse(single_cinder_volume_template)
    stack = utils.parse_stack(t, stack_name=stack_name)
    self.patchobject(stack['volume'], '_store_config_default_properties')
    scheduler.TaskRunner(stack.create)()
    self.assertEqual((stack.CREATE, stack.COMPLETE), stack.state)
    scheduler.TaskRunner(stack.snapshot, None)()
    self.assertEqual((stack.SNAPSHOT, stack.COMPLETE), stack.state)
    data = stack.prepare_abandon()
    fake_snapshot = collections.namedtuple('Snapshot', ('data', 'stack_id'))(data, stack.id)
    stack.restore(fake_snapshot)
    self.assertEqual(stack_final_status, stack.state)
    self.cinder_fc.volumes.create.assert_called_once_with(size=1, availability_zone=None, description='test_description', name='test_name', metadata={})
    self.cinder_fc.backups.create.assert_called_once_with(fv.id, force=True)
    self.cinder_fc.backups.get.assert_called_once_with(fb.id)
    self.cinder_fc.restores.restore.assert_called_once_with('backup-123', 'vol-123')
    self.assertEqual(3, self.cinder_fc.volumes.get.call_count)