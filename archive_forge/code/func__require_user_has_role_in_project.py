from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _require_user_has_role_in_project(self, roles, user_id, project_id):
    user_roles = self._get_user_roles(user_id, project_id)
    for role in roles:
        if role['id'] not in user_roles:
            raise exception.RoleAssignmentNotFound(role_id=role['id'], actor_id=user_id, target_id=project_id)