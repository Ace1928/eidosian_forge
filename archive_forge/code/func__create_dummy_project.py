import os
import unittest
import fixtures
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_exceptions
from openstackclient.tests.functional import base
def _create_dummy_project(self, add_clean_up=True):
    project_name = data_utils.rand_name('TestProject')
    project_description = data_utils.rand_name('description')
    raw_output = self.openstack('project create --description %(description)s --enable %(name)s' % {'description': project_description, 'name': project_name})
    project = self.parse_show_as_object(raw_output)
    if add_clean_up:
        self.addCleanup(self.openstack, 'project delete %s' % project['id'])
    items = self.parse_show(raw_output)
    self.assert_show_fields(items, self.PROJECT_FIELDS)
    return project_name