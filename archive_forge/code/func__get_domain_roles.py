from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
def _get_domain_roles(self):
    roles = []
    domain_roles = PROVIDERS.assignment_api.get_roles_for_user_and_domain(self.user_id, self.domain_id)
    for role_id in domain_roles:
        role = PROVIDERS.role_api.get_role(role_id)
        roles.append({'id': role['id'], 'name': role['name']})
    return roles