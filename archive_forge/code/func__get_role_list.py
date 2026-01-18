from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _get_role_list(self, app_cred_roles):
    roles = []
    for role in app_cred_roles:
        roles.append(PROVIDERS.role_api.get_role(role['id']))
    return roles