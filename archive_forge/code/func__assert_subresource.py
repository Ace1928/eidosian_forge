import importlib.util
import os
from neutron_lib.api import definitions
from neutron_lib.api.definitions import base
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.tests import _base as test_base
def _assert_subresource(self, subresource):
    self.assertIn(self.subresource_map[subresource]['parent']['collection_name'], base.KNOWN_RESOURCES + self.extension_resources, 'Sub-resource parent is unknown, check for typos.')
    self.assertIn('member_name', self.subresource_map[subresource]['parent'], 'Incorrect parent definition, check for typos.')
    self.assertParams(self.subresource_map[subresource]['parameters'])