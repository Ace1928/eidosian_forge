import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
class ProjectsTestMixin(object):

    def check_project(self, project, project_ref=None):
        self.assertIsNotNone(project.id)
        self.assertIn('self', project.links)
        self.assertIn('/projects/' + project.id, project.links['self'])
        if project_ref:
            self.assertEqual(project_ref['name'], project.name)
            self.assertEqual(project_ref['domain'], project.domain_id)
            self.assertEqual(project_ref['enabled'], project.enabled)
            if hasattr(project_ref, 'description'):
                self.assertEqual(project_ref['description'], project.description)
            if hasattr(project_ref, 'parent'):
                self.assertEqual(project_ref['parent'], project.parent)
        else:
            self.assertIsNotNone(project.name)
            self.assertIsNotNone(project.domain_id)
            self.assertIsNotNone(project.enabled)