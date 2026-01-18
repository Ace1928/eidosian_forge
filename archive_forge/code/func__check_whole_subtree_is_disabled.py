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
def _check_whole_subtree_is_disabled(self, project_id, subtree_list=None):
    if not subtree_list:
        subtree_list = self.list_projects_in_subtree(project_id)
    subtree_enabled = [ref.get('enabled', True) for ref in subtree_list]
    return not any(subtree_enabled)