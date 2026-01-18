from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
def _get_federated_roles(self):
    roles = []
    group_ids = [group['id'] for group in self.federated_groups]
    federated_roles = PROVIDERS.assignment_api.get_roles_for_groups(group_ids, self.project_id, self.domain_id)
    for group_id in group_ids:
        group_roles = PROVIDERS.assignment_api.list_system_grants_for_group(group_id)
        for role in group_roles:
            federated_roles.append(role)
    user_roles = PROVIDERS.assignment_api.list_system_grants_for_user(self.user_id)
    for role in user_roles:
        federated_roles.append(role)
    if self.domain_id:
        domain_roles = PROVIDERS.assignment_api.get_roles_for_user_and_domain(self.user_id, self.domain_id)
        for role in domain_roles:
            federated_roles.append(role)
    if self.project_id:
        project_roles = PROVIDERS.assignment_api.get_roles_for_user_and_project(self.user_id, self.project_id)
        for role in project_roles:
            federated_roles.append(role)
    for role in federated_roles:
        if not isinstance(role, dict):
            role = PROVIDERS.role_api.get_role(role)
        if role not in roles:
            roles.append(role)
    return roles