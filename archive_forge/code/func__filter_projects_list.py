from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def _filter_projects_list(self, projects_list, user_id):
    user_projects = PROVIDERS.assignment_api.list_projects_for_user(user_id)
    user_projects_ids = set([proj['id'] for proj in user_projects])
    return [proj for proj in projects_list if proj['id'] in user_projects_ids]