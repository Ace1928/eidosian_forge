import warnings
from twisted.trial.unittest import TestCase
def _initializedOnceTest(self, container, constantName):
    """
        Assert that C{container._enumerants} does not change as a side-effect
        of one of its attributes being accessed.

        @param container: A L{_ConstantsContainer} subclass which will be
            tested.
        @param constantName: The name of one of the constants which is an
            attribute of C{container}.
        """
    first = container._enumerants
    getattr(container, constantName)
    second = container._enumerants
    self.assertIs(first, second)