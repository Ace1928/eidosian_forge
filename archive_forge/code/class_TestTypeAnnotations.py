import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class TestTypeAnnotations(unittest.TestCase):
    """Test using Interfaces in type annotations."""

    def test___or__(self):
        from typing import Optional
        from typing import Union
        from zope.interface import Interface

        class I1(Interface):
            pass

        class I2(Interface):
            pass

        class B:
            a: I1 | None
            b: I1 | I2
        self.assertEqual(B.__annotations__, {'a': Optional[I1], 'b': Union[I1, I2]})

    def test___ror__(self):
        from typing import Optional
        from typing import Union
        from zope.interface import Interface

        class I1(Interface):
            pass

        class A:
            pass

        class B:
            a: None | I1
            b: A | I1
        self.assertEqual(B.__annotations__, {'a': Optional[I1], 'b': Union[A, I1]})