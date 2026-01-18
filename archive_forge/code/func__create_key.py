from castellan.common import objects
from castellan.common.objects import private_key
from castellan.tests import base
from castellan.tests import utils
def _create_key(self):
    return private_key.PrivateKey(self.algorithm, self.bit_length, self.encoded, self.name, self.created, consumers=self.consumers)