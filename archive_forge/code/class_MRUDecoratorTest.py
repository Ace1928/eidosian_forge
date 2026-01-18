import unittest
import cachetools.func
class MRUDecoratorTest(unittest.TestCase, DecoratorTestMixin):
    DECORATOR = staticmethod(cachetools.func.mru_cache)