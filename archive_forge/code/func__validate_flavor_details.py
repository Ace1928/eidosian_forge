import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def _validate_flavor_details(self, flavor_details, server_details):
    flavor_key_mapping = {'OS-FLV-EXT-DATA:ephemeral': 'flavor:ephemeral', 'disk': 'flavor:disk', 'extra_specs': 'flavor:extra_specs', 'name': 'flavor:original_name', 'ram': 'flavor:ram', 'swap': 'flavor:swap', 'vcpus': 'flavor:vcpus'}
    for key in flavor_key_mapping:
        flavor_val = self._get_value_from_the_table(flavor_details, key)
        server_flavor_val = self._get_value_from_the_table(server_details, flavor_key_mapping[key])
        if key == 'swap' and flavor_val == '':
            flavor_val = '0'
        self.assertEqual(flavor_val, server_flavor_val)