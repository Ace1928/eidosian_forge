import unittest
from traits.api import HasTraits
class TestPythonProperty(unittest.TestCase):

    def test_read_only_property(self):
        model = Model()
        self.assertEqual(model.read_only, 1729)
        with self.assertRaises(AttributeError):
            model.read_only = 2034
        with self.assertRaises(AttributeError):
            del model.read_only

    def test_read_write_property(self):
        model = Model()
        self.assertEqual(model.value, 0)
        model.value = 23
        self.assertEqual(model.value, 23)
        model.value = 77
        self.assertEqual(model.value, 77)
        with self.assertRaises(AttributeError):
            del model.value