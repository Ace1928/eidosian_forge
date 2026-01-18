from keystone.assignment.backends import base
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
def _get_domain_assignment_types(self):
    return [AssignmentType.USER_DOMAIN, AssignmentType.GROUP_DOMAIN]