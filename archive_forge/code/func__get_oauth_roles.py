from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
def _get_oauth_roles(self):
    roles = []
    access_token_roles = self.access_token['role_ids']
    access_token_roles = [{'role_id': r} for r in jsonutils.loads(access_token_roles)]
    effective_access_token_roles = PROVIDERS.assignment_api.add_implied_roles(access_token_roles)
    user_roles = [r['id'] for r in self._get_project_roles()]
    for role in effective_access_token_roles:
        if role['role_id'] in user_roles:
            role = PROVIDERS.role_api.get_role(role['role_id'])
            roles.append({'id': role['id'], 'name': role['name']})
    return roles