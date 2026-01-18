import unittest
def _make_complex_diamond(self, base):
    O = base

    class F(O):
        pass

    class E(O):
        pass

    class D(O):
        pass

    class C(D, F):
        pass

    class B(D, E):
        pass

    class A(B, C):
        pass
    if hasattr(A, 'mro'):
        self.assertEqual(A.mro(), self._callFUT(A))
    return A