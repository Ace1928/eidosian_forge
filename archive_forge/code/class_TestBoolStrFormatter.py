from openstack import format
from openstack.tests.unit import base
class TestBoolStrFormatter(base.TestCase):

    def test_deserialize(self):
        self.assertTrue(format.BoolStr.deserialize(True))
        self.assertTrue(format.BoolStr.deserialize('True'))
        self.assertTrue(format.BoolStr.deserialize('TRUE'))
        self.assertTrue(format.BoolStr.deserialize('true'))
        self.assertFalse(format.BoolStr.deserialize(False))
        self.assertFalse(format.BoolStr.deserialize('False'))
        self.assertFalse(format.BoolStr.deserialize('FALSE'))
        self.assertFalse(format.BoolStr.deserialize('false'))
        self.assertRaises(ValueError, format.BoolStr.deserialize, None)
        self.assertRaises(ValueError, format.BoolStr.deserialize, '')
        self.assertRaises(ValueError, format.BoolStr.deserialize, 'INVALID')