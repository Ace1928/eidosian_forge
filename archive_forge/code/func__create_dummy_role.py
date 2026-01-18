import os
import unittest
import fixtures
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_exceptions
from openstackclient.tests.functional import base
def _create_dummy_role(self, add_clean_up=True):
    role_name = data_utils.rand_name('TestRole')
    raw_output = self.openstack('role create %s' % role_name)
    role = self.parse_show_as_object(raw_output)
    if add_clean_up:
        self.addCleanup(self.openstack, 'role delete %s' % role['id'])
    items = self.parse_show(raw_output)
    self.assert_show_fields(items, self.ROLE_FIELDS)
    self.assertEqual(role_name, role['name'])
    return role_name