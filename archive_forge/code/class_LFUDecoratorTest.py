import unittest
import cachetools.func
class LFUDecoratorTest(unittest.TestCase, DecoratorTestMixin):
    DECORATOR = staticmethod(cachetools.func.lfu_cache)