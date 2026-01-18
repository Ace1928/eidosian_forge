from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import role_assignments
from heat.engine import support
from heat.engine import translation
def _add_user_to_groups(self, user_id, groups):
    if groups is not None:
        for group_id in groups:
            self.client().users.add_to_group(user_id, group_id)