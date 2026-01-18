import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def _find_network_in_table(self, table):
    for line in table.split('\n'):
        if '|' in line:
            l_property, l_value = line.split('|')[1:3]
            if ' network' in l_property.strip():
                return ' '.join(l_property.strip().split()[:-1])