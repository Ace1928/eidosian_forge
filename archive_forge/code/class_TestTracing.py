from io import StringIO
import logging
import unittest
from numba.core import tracing
class TestTracing(unittest.TestCase):

    def __init__(self, *args):
        super(TestTracing, self).__init__(*args)

    def setUp(self):
        self.capture = CapturedTrace()

    def tearDown(self):
        del self.capture

    def test_method(self):
        with self.capture:
            Class().method('foo', bar='baz')
        self.assertEqual(self.capture.getvalue(), ">> Class.method(self=<Class instance>, some='foo', other='value', bar='baz')\n" + '<< Class.method\n')

    def test_class_method(self):
        with self.capture:
            Class.class_method()
        self.assertEqual(self.capture.getvalue(), ">> Class.class_method(cls=<class 'Class'>)\n" + '<< Class.class_method\n')

    def test_static_method(self):
        with self.capture:
            Class.static_method()
        self.assertEqual(self.capture.getvalue(), '>> static_method()\n' + '<< static_method\n')

    def test_property(self):
        with self.capture:
            test = Class()
            test.test = 1
            assert 1 == test.test
        self.assertEqual(self.capture.getvalue(), '>> Class._test_set(self=<Class instance>, value=1)\n' + '<< Class._test_set\n' + '>> Class._test_get(self=<Class instance>)\n' + '<< Class._test_get -> 1\n')

    def test_function(self):
        with self.capture:
            test(5, 5)
            test(5, 5, False)
        self.assertEqual(self.capture.getvalue(), '>> test(x=5, y=5, z=True)\n' + '<< test -> 10\n' + '>> test(x=5, y=5, z=False)\n' + '<< test -> 25\n')

    @unittest.skip('recursive decoration not yet implemented')
    def test_injected(self):
        with self.capture:
            tracing.trace(Class2, recursive=True)
            Class2.class_method()
            Class2.static_method()
            test = Class2()
            test.test = 1
            assert 1 == test.test
            test.method()
            self.assertEqual(self.capture.getvalue(), ">> Class2.class_method(cls=<type 'Class2'>)\n" + '<< Class2.class_method\n>> static_method()\n<< static_method\n')