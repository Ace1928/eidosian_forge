import uuid
from oslo_log import log
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.server import backends
def _bootstrap_admin_role(self):
    role = self._ensure_role_exists(self.admin_role_name)
    self.admin_role_id = role['id']
    self._ensure_implied_role(self.admin_role_id, self.manager_role_id)
    try:
        PROVIDERS.role_api.delete_implied_role(self.admin_role_id, self.member_role_id)
    except exception.ImpliedRoleNotFound:
        pass