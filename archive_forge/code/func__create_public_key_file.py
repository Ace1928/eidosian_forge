import tempfile
from tempest.lib import exceptions
from novaclient.tests.functional import base
from novaclient.tests.functional.v2 import fake_crypto
def _create_public_key_file(self, public_key):
    pubfile = tempfile.mkstemp()[1]
    with open(pubfile, 'w') as f:
        f.write(public_key)
    return pubfile