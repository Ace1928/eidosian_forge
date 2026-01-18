from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def _handle_project_spec(self, test_data, domain_id, project_spec, parent_id=None):
    """Handle the creation of a project or hierarchy of projects.

        project_spec may either be a count of the number of projects to
        create, or it may be a list of the form:

        [{'project': project_spec}, {'project': project_spec}, ...]

        This method is called recursively to handle the creation of a
        hierarchy of projects.

        """

    def _create_project(domain_id, parent_id):
        new_project = unit.new_project_ref(domain_id=domain_id, parent_id=parent_id)
        new_project = PROVIDERS.resource_api.create_project(new_project['id'], new_project)
        return new_project
    if isinstance(project_spec, list):
        for this_spec in project_spec:
            self._handle_project_spec(test_data, domain_id, this_spec, parent_id=parent_id)
    elif isinstance(project_spec, dict):
        new_proj = _create_project(domain_id, parent_id)
        test_data['projects'].append(new_proj)
        self._handle_project_spec(test_data, domain_id, project_spec['project'], parent_id=new_proj['id'])
    else:
        for _ in range(project_spec):
            test_data['projects'].append(_create_project(domain_id, parent_id))