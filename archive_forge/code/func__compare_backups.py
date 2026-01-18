from openstack.block_storage.v3 import backup
from openstack.tests.unit import base
def _compare_backups(self, exp, real):
    self.assertDictEqual(backup.Backup(**exp).to_dict(computed=False), real.to_dict(computed=False))