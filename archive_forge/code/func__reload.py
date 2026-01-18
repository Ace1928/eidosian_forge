import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
def _reload(self):
    if not self._container_ref:
        raise AttributeError('container_ref not set, cannot reload data.')
    LOG.debug('Getting container - Container href: {0}'.format(self._container_ref))
    uuid_ref = base.calculate_uuid_ref(self._container_ref, self._entity)
    try:
        response = self._api.get(uuid_ref)
    except AttributeError:
        raise LookupError('Container {0} could not be found.'.format(self._container_ref))
    self._name = response.get('name')
    self._consumers = response.get('consumers', [])
    created = response.get('created')
    updated = response.get('updated')
    self._created = parse_isotime(created) if created else None
    self._updated = parse_isotime(updated) if updated else None
    self._status = response.get('status')