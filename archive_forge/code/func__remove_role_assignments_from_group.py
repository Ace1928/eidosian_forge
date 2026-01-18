from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def _remove_role_assignments_from_group(self, group_id, role_assignments, current_assignments):
    for role_assignment in self._normalize_to_id(role_assignments):
        if role_assignment in current_assignments:
            if role_assignment.get(self.PROJECT) is not None:
                self.client().roles.revoke(role=role_assignment.get(self.ROLE), project=role_assignment.get(self.PROJECT), group=group_id)
            elif role_assignment.get(self.DOMAIN) is not None:
                self.client().roles.revoke(role=role_assignment.get(self.ROLE), domain=role_assignment.get(self.DOMAIN), group=group_id)