import unittest
import cachetools.func
class LRUDecoratorTest(unittest.TestCase, DecoratorTestMixin):
    DECORATOR = staticmethod(cachetools.func.lru_cache)