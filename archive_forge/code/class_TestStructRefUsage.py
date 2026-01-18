import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.experimental import structref
from numba.tests.support import skip_unless_scipy
@skip_unless_scipy
class TestStructRefUsage(unittest.TestCase):

    def test_type_definition(self):
        np.random.seed(0)
        buf = []

        def print(*args):
            buf.append(args)
        alice = MyStruct('Alice', vector=np.random.random(3))

        @njit
        def make_bob():
            bob = MyStruct('unnamed', vector=np.zeros(3))
            bob.name = 'Bob'
            bob.vector = np.random.random(3)
            return bob
        bob = make_bob()
        print(f'{alice.name}: {alice.vector}')
        print(f'{bob.name}: {bob.vector}')

        @njit
        def distance(a, b):
            return np.linalg.norm(a.vector - b.vector)
        print(distance(alice, bob))
        self.assertEqual(len(buf), 3)

    def test_overload_method(self):
        from numba.core.extending import overload_method
        from numba.core.errors import TypingError

        @overload_method(MyStructType, 'distance')
        def ol_distance(self, other):
            if not isinstance(other, MyStructType):
                raise TypingError(f'*other* must be a {MyStructType}; got {other}')

            def impl(self, other):
                return np.linalg.norm(self.vector - other.vector)
            return impl

        @njit
        def test():
            alice = MyStruct('Alice', vector=np.random.random(3))
            bob = MyStruct('Bob', vector=np.random.random(3))
            return alice.distance(bob)
        self.assertIsInstance(test(), float)