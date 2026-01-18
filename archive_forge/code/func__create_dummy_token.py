import os
import unittest
import fixtures
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_exceptions
from openstackclient.tests.functional import base
def _create_dummy_token(self, add_clean_up=True):
    raw_output = self.openstack('token issue')
    token = self.parse_show_as_object(raw_output)
    if add_clean_up:
        self.addCleanup(self.openstack, 'token revoke %s' % token['id'])
    items = self.parse_show(raw_output)
    self.assert_show_fields(items, self.TOKEN_FIELDS)
    return token['id']