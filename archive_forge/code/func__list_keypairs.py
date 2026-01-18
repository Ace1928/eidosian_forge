import tempfile
from tempest.lib import exceptions
from novaclient.tests.functional import base
from novaclient.tests.functional.v2 import fake_crypto
def _list_keypairs(self):
    return self.nova('keypair-list')