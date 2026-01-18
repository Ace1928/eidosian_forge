import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def _boot_server_with_description(self):
    descr = 'Some words about this test VM.'
    server = self._create_server(description=descr)
    self.assertEqual(descr, server.description)
    return (server, descr)