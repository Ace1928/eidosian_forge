import warnings
from twisted.trial.unittest import TestCase
def _notInstantiableTest(self, name, cls):
    """
        Assert that an attempt to instantiate the constants class raises
        C{TypeError}.

        @param name: A C{str} giving the name of the constants collection.
        @param cls: The constants class to test.
        """
    exc = self.assertRaises(TypeError, cls)
    self.assertEqual(name + ' may not be instantiated.', str(exc))