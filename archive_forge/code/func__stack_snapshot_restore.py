import os
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib import exceptions as tempest_exc
def _stack_snapshot_restore(self, id, snapshot_id):
    cmd = 'stack snapshot restore ' + id + ' ' + snapshot_id
    self.openstack(cmd)