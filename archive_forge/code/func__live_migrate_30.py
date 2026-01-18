import typing as ty
from openstack.common import metadata
from openstack.common import tag
from openstack.compute.v2 import flavor
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.image.v2 import image
from openstack import resource
from openstack import utils
def _live_migrate_30(self, session, host, force, block_migration):
    microversion = '2.30'
    body = {'host': None}
    if block_migration is None:
        block_migration = 'auto'
    body['block_migration'] = block_migration
    if host:
        body['host'] = host
        if force:
            body['force'] = force
    self._action(session, {'os-migrateLive': body}, microversion=microversion)