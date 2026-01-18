import uuid
from oslo_log import log
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.server import backends
def _bootstrap_project_role_assignment(self):
    try:
        PROVIDERS.assignment_api.add_role_to_user_and_project(user_id=self.admin_user_id, project_id=self.project_id, role_id=self.admin_role_id)
        LOG.info('Granted role %(role)s on project %(project)s to user %(username)s.', {'role': self.admin_role_name, 'project': self.project_name, 'username': self.admin_username})
    except exception.Conflict:
        LOG.info('User %(username)s already has role %(role)s on project %(project)s.', {'username': self.admin_username, 'role': self.admin_role_name, 'project': self.project_name})