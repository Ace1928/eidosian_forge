import typing as ty
from openstack.common import metadata
from openstack.common import tag
from openstack.compute.v2 import flavor
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.image.v2 import image
from openstack import resource
from openstack import utils
def _live_migrate_25(self, session, host, force, block_migration):
    microversion = '2.25'
    body = {'host': None}
    if block_migration is None:
        block_migration = 'auto'
    body['block_migration'] = block_migration
    if host:
        body['host'] = host
        if not force:
            raise ValueError("Live migration on this cloud implies 'force' if the 'host' option has been given and it is not possible to disable. It is recommended to not use 'host' at all on this cloud as it is inherently unsafe, but if it is unavoidable, please supply 'force=True' so that it is clear you understand the risks.")
    self._action(session, {'os-migrateLive': body}, microversion=microversion)