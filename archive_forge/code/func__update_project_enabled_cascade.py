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
def _update_project_enabled_cascade(self, project_id, enabled):
    subtree = self.list_projects_in_subtree(project_id)
    subtree_to_update = [child for child in subtree if child['enabled'] != enabled]
    for child in subtree_to_update:
        child['enabled'] = enabled
        if not enabled:
            notifications.Audit.disabled(self._PROJECT, child['id'], public=False)
        self.driver.update_project(child['id'], child)