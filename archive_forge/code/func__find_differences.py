from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def _find_differences(self, updated_prps, stored_prps):
    updated_role_project_assignments = []
    updated_role_domain_assignments = []
    for role_assignment in updated_prps or []:
        if role_assignment.get(self.PROJECT) is not None:
            updated_role_project_assignments.append('%s:%s' % (role_assignment[self.ROLE], role_assignment[self.PROJECT]))
        elif role_assignment.get(self.DOMAIN) is not None:
            updated_role_domain_assignments.append('%s:%s' % (role_assignment[self.ROLE], role_assignment[self.DOMAIN]))
    stored_role_project_assignments = []
    stored_role_domain_assignments = []
    for role_assignment in stored_prps or []:
        if role_assignment.get(self.PROJECT) is not None:
            stored_role_project_assignments.append('%s:%s' % (role_assignment[self.ROLE], role_assignment[self.PROJECT]))
        elif role_assignment.get(self.DOMAIN) is not None:
            stored_role_domain_assignments.append('%s:%s' % (role_assignment[self.ROLE], role_assignment[self.DOMAIN]))
    new_role_assignments = []
    removed_role_assignments = []
    for item in set(updated_role_project_assignments) - set(stored_role_project_assignments):
        new_role_assignments.append({self.ROLE: item[:item.find(':')], self.PROJECT: item[item.find(':') + 1:]})
    for item in set(updated_role_domain_assignments) - set(stored_role_domain_assignments):
        new_role_assignments.append({self.ROLE: item[:item.find(':')], self.DOMAIN: item[item.find(':') + 1:]})
    for item in set(stored_role_project_assignments) - set(updated_role_project_assignments):
        removed_role_assignments.append({self.ROLE: item[:item.find(':')], self.PROJECT: item[item.find(':') + 1:]})
    for item in set(stored_role_domain_assignments) - set(updated_role_domain_assignments):
        removed_role_assignments.append({self.ROLE: item[:item.find(':')], self.DOMAIN: item[item.find(':') + 1:]})
    return (new_role_assignments, removed_role_assignments)