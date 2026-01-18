import unittest
from traits.testing.nose_tools import deprecated, performance, skip
class TestNoseTools(unittest.TestCase):

    def test_deprecated_deprecated(self):
        with self.assertWarns(DeprecationWarning) as cm:

            @deprecated
            def some_func():
                pass
        self.assertIn('test_nose_tools', cm.filename)

    def test_performance_deprecated(self):
        with self.assertWarns(DeprecationWarning) as cm:

            @performance
            def some_func():
                pass
        self.assertIn('test_nose_tools', cm.filename)

    def test_skip_deprecated(self):
        with self.assertWarns(DeprecationWarning) as cm:

            @skip
            def some_func():
                pass
        self.assertIn('test_nose_tools', cm.filename)