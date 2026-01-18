from castellan.common import objects
from castellan.common.objects import x_509
from castellan.tests import base
from castellan.tests import utils
def _create_cert(self):
    return x_509.X509(self.data, self.name, self.created, consumers=self.consumers)