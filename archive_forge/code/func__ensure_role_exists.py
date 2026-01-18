import uuid
from oslo_log import log
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.server import backends
def _ensure_role_exists(self, role_name):
    try:
        role_id = uuid.uuid4().hex
        role = {'name': role_name, 'id': role_id}
        if self.immutable_roles:
            role['options'] = {'immutable': True}
        role = PROVIDERS.role_api.create_role(role_id, role)
        LOG.info('Created role %s', role_name)
        if not self.immutable_roles:
            LOG.warning("Role %(role)s was created as a mutable role. It is recommended to make this role immutable by adding the 'immutable' resource option to this role, or re-running this command without --no-immutable-role.", {'role': role_name})
        return role
    except exception.Conflict:
        LOG.info('Role %s exists, skipping creation.', role_name)
        hints = driver_hints.Hints()
        hints.add_filter('name', role_name)
        hints.add_filter('domain_id', None)
        return PROVIDERS.role_api.list_roles(hints)[0]