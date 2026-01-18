import warnings
from twisted.trial.unittest import TestCase
class _ConstantsTestsMixin:
    """
    Mixin defining test helpers common to multiple types of constants
    collections.
    """

    def _notInstantiableTest(self, name, cls):
        """
        Assert that an attempt to instantiate the constants class raises
        C{TypeError}.

        @param name: A C{str} giving the name of the constants collection.
        @param cls: The constants class to test.
        """
        exc = self.assertRaises(TypeError, cls)
        self.assertEqual(name + ' may not be instantiated.', str(exc))

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