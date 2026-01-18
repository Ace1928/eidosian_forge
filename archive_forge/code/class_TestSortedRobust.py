from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.sorting import sorted_robust, _robust_sort_keyfcn
class TestSortedRobust(unittest.TestCase):

    def test_sorted_robust(self):
        a = sorted_robust([3, 2, 1])
        self.assertEqual(a, [1, 2, 3])
        a = sorted_robust([3, 2.1, 1])
        self.assertEqual(a, [1, 2.1, 3])
        a = sorted_robust([3, '2', 1])
        self.assertEqual(a, [1, 3, '2'])
        a = sorted_robust([('str1', 'str1'), (1, 'str2')])
        self.assertEqual(a, [(1, 'str2'), ('str1', 'str1')])
        a = sorted_robust([((1,), 'str2'), ('str1', 'str1')])
        self.assertEqual(a, [('str1', 'str1'), ((1,), 'str2')])
        a = sorted_robust([('str1', 'str1'), ((1,), 'str2')])
        self.assertEqual(a, [('str1', 'str1'), ((1,), 'str2')])

    def test_user_key(self):
        sorted_robust([(('10_1', 2), None), ((10, 2), None)], key=lambda x: x[0])

    def test_unknown_types(self):
        orig = [LikeFloat(4), Comparable('hello'), LikeFloat(1), 2.0, Comparable('world'), ToStr(1), NoStr('bogus'), ToStr('a'), ToStr('A'), 3]
        ref = [orig[i] for i in (1, 4, 6, 5, 8, 7, 2, 3, 9, 0)]
        ans = sorted_robust(orig)
        self.assertEqual(len(orig), len(ans))
        for _r, _a in zip(ref, ans):
            self.assertIs(_r, _a)
        self.assertEqual(_robust_sort_keyfcn._typemap[LikeFloat], (1, float.__name__))
        self.assertEqual(_robust_sort_keyfcn._typemap[Comparable], (1, Comparable.__name__))
        self.assertEqual(_robust_sort_keyfcn._typemap[ToStr], (2, ToStr.__name__))
        self.assertEqual(_robust_sort_keyfcn._typemap[NoStr], (3, NoStr.__name__))