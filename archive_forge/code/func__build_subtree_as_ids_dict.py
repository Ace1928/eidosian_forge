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
def _build_subtree_as_ids_dict(self, project_id, subtree_by_parent):

    def traverse_subtree_hierarchy(project_id):
        children = subtree_by_parent.get(project_id)
        if not children:
            return None
        children_ids = {}
        for child in children:
            children_ids[child['id']] = traverse_subtree_hierarchy(child['id'])
        return children_ids
    return traverse_subtree_hierarchy(project_id)