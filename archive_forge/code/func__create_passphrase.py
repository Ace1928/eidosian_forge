from castellan.common import objects
from castellan.common.objects import passphrase
from castellan.tests import base
def _create_passphrase(self):
    return passphrase.Passphrase(self.passphrase_data, self.name, self.created, consumers=self.consumers)