from castellan.common import exception
from castellan.common import objects
from castellan.tests import base
class ManagedObjectFromDictTestCase(base.TestCase):

    def test_invalid_dict(self):
        self.assertRaises(exception.InvalidManagedObjectDictError, objects.from_dict, {})

    def test_unknown_type(self):
        self.assertRaises(exception.UnknownManagedObjectTypeError, objects.from_dict, {'type': 'non-existing-managed-object-type'})