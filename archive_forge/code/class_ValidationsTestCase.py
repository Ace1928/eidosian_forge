import io
import sys
from unittest import mock
from urllib import parse
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils as test_utils
from novaclient import utils
from novaclient.v2 import servers
class ValidationsTestCase(test_utils.TestCase):

    def test_validate_flavor_metadata_keys_with_valid_keys(self):
        valid_keys = ['key1', 'month.price', 'I-Am:AK-ey.01-', 'spaces and _']
        utils.validate_flavor_metadata_keys(valid_keys)

    def test_validate_flavor_metadata_keys_with_invalid_keys(self):
        invalid_keys = ['/1', '?1', '%1', '<', '>', '\x01']
        for key in invalid_keys:
            try:
                utils.validate_flavor_metadata_keys([key])
                self.fail('Invalid key passed validation: %s' % key)
            except exceptions.CommandError as ce:
                self.assertIn(key, str(ce))