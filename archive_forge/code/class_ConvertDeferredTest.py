import unittest
from tornado.testing import AsyncTestCase, gen_test
@skipIfNoTwisted
class ConvertDeferredTest(AsyncTestCase):

    @gen_test
    def test_success(self):

        @inlineCallbacks
        def fn():
            if False:
                yield
            returnValue(42)
        res = (yield fn())
        self.assertEqual(res, 42)

    @gen_test
    def test_failure(self):

        @inlineCallbacks
        def fn():
            if False:
                yield
            1 / 0
        with self.assertRaises(ZeroDivisionError):
            yield fn()