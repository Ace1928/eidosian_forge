import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def _boot_server_with_tags(self, tags=['t1', 't2']):
    uuid = self._create_server().id
    self.client.servers.set_tags(uuid, tags)
    return uuid