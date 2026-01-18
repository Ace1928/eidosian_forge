import unittest
import cachetools
import cachetools.keys
class NoneWrapperTest(unittest.TestCase):

    def func(self, *args, **kwargs):
        return args + tuple(kwargs.items())

    def test_decorator(self):
        wrapper = cachetools.cached(None)(self.func)
        self.assertEqual(wrapper.__wrapped__, self.func)
        self.assertEqual(wrapper(0), (0,))
        self.assertEqual(wrapper(1), (1,))
        self.assertEqual(wrapper(1, foo='bar'), (1, ('foo', 'bar')))