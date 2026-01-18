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
def _build_parents_as_ids_dict(self, project, parents_by_id):

    def traverse_parents_hierarchy(project):
        parent_id = project.get('parent_id')
        if not parent_id:
            return None
        parent = parents_by_id[parent_id]
        return {parent_id: traverse_parents_hierarchy(parent)}
    return traverse_parents_hierarchy(project)