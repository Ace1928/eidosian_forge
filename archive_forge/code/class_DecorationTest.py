import unittest as pyunit
from twisted.python.util import mergeFunctionMetadata
from twisted.trial import unittest
class DecorationTest(unittest.SynchronousTestCase):

    def test_badDecorator(self) -> None:
        """
        This test method is decorated in a way that gives it a confusing name
        that collides with another method.
        """
    test_badDecorator = badDecorator(test_badDecorator)

    def test_goodDecorator(self) -> None:
        """
        This test method is decorated in a way that preserves its name.
        """
    test_goodDecorator = goodDecorator(test_goodDecorator)

    def renamedDecorator(self) -> None:
        """
        This is secretly a test method and will be decorated and then renamed so
        test discovery can find it.
        """
    test_renamedDecorator = goodDecorator(renamedDecorator)

    def nameCollision(self) -> None:
        """
        This isn't a test, it's just here to collide with tests.
        """