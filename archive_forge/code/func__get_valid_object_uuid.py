from oslo_utils import uuidutils
from castellan.common import exception
from castellan.common.objects import opaque_data
from castellan.common.objects import passphrase
from castellan.common.objects import private_key
from castellan.common.objects import public_key
from castellan.common.objects import symmetric_key
from castellan.common.objects import x_509
from castellan.tests import utils
def _get_valid_object_uuid(self, managed_object):
    object_uuid = self.key_mgr.store(self.ctxt, managed_object)
    self.assertIsNotNone(object_uuid)
    return object_uuid