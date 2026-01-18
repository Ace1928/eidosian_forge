from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def _add_projects_access(self, projects):
    for project in projects:
        project_id = self.client_plugin('keystone').get_project_id(project)
        self.client().volume_type_access.add_project_access(self.resource_id, project_id)