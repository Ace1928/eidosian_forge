import unittest
class Test__mergeOrderings(unittest.TestCase):

    def _callFUT(self, orderings):
        from zope.interface.ro import _legacy_mergeOrderings
        return _legacy_mergeOrderings(orderings)

    def test_empty(self):
        self.assertEqual(self._callFUT([]), [])

    def test_single(self):
        self.assertEqual(self._callFUT(['a', 'b', 'c']), ['a', 'b', 'c'])

    def test_w_duplicates(self):
        self.assertEqual(self._callFUT([['a'], ['b', 'a']]), ['b', 'a'])

    def test_suffix_across_multiple_duplicates(self):
        O1 = ['x', 'y', 'z']
        O2 = ['q', 'z']
        O3 = [1, 3, 5]
        O4 = ['z']
        self.assertEqual(self._callFUT([O1, O2, O3, O4]), ['x', 'y', 'q', 1, 3, 5, 'z'])