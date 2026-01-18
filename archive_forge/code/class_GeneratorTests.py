from greenlet import greenlet
from . import TestCase
class GeneratorTests(TestCase):

    def test_generator(self):
        seen = []

        def g(n):
            for i in range(n):
                seen.append(i)
                Yield(i)
        g = generator(g)
        for _ in range(3):
            for j in g(5):
                seen.append(j)
        self.assertEqual(seen, 3 * [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])