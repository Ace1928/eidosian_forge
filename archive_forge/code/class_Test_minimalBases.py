import sys
import unittest
import sys
class Test_minimalBases(unittest.TestCase):

    def _callFUT(self, klasses):
        from zope.interface.advice import minimalBases
        return minimalBases(klasses)

    def test_empty(self):
        self.assertEqual(self._callFUT([]), [])

    def test_w_newstyle_meta(self):
        self.assertEqual(self._callFUT([type]), [type])

    def test_w_newstyle_class(self):

        class C:
            pass
        self.assertEqual(self._callFUT([C]), [C])

    def test_simple_hierarchy_skips_implied(self):

        class A:
            pass

        class B(A):
            pass

        class C(B):
            pass

        class D:
            pass
        self.assertEqual(self._callFUT([A, B, C]), [C])
        self.assertEqual(self._callFUT([A, C]), [C])
        self.assertEqual(self._callFUT([B, C]), [C])
        self.assertEqual(self._callFUT([A, B]), [B])
        self.assertEqual(self._callFUT([D, B, D]), [B, D])

    def test_repeats_kicked_to_end_of_queue(self):

        class A:
            pass

        class B:
            pass
        self.assertEqual(self._callFUT([A, B, A]), [B, A])