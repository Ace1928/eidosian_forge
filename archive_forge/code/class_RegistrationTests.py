from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
class RegistrationTests(RegistryUsingMixin, unittest.SynchronousTestCase):
    """
    Tests for adapter registration.
    """

    def _registerAdapterForClassOrInterface(self, original):
        """
        Register an adapter with L{components.registerAdapter} for the given
        class or interface and verify that the adapter can be looked up with
        L{components.getAdapterFactory}.
        """
        adapter = lambda o: None
        components.registerAdapter(adapter, original, ITest)
        self.assertIs(components.getAdapterFactory(original, ITest, None), adapter)

    def test_registerAdapterForClass(self):
        """
        Test that an adapter from a class can be registered and then looked
        up.
        """

        class TheOriginal:
            pass
        return self._registerAdapterForClassOrInterface(TheOriginal)

    def test_registerAdapterForInterface(self):
        """
        Test that an adapter from an interface can be registered and then
        looked up.
        """
        return self._registerAdapterForClassOrInterface(ITest2)

    def _duplicateAdapterForClassOrInterface(self, original):
        """
        Verify that L{components.registerAdapter} raises L{ValueError} if the
        from-type/interface and to-interface pair is not unique.
        """
        firstAdapter = lambda o: False
        secondAdapter = lambda o: True
        components.registerAdapter(firstAdapter, original, ITest)
        self.assertRaises(ValueError, components.registerAdapter, secondAdapter, original, ITest)
        self.assertIs(components.getAdapterFactory(original, ITest, None), firstAdapter)

    def test_duplicateAdapterForClass(self):
        """
        Test that attempting to register a second adapter from a class
        raises the appropriate exception.
        """

        class TheOriginal:
            pass
        return self._duplicateAdapterForClassOrInterface(TheOriginal)

    def test_duplicateAdapterForInterface(self):
        """
        Test that attempting to register a second adapter from an interface
        raises the appropriate exception.
        """
        return self._duplicateAdapterForClassOrInterface(ITest2)

    def _duplicateAdapterForClassOrInterfaceAllowed(self, original):
        """
        Verify that when C{components.ALLOW_DUPLICATES} is set to C{True}, new
        adapter registrations for a particular from-type/interface and
        to-interface pair replace older registrations.
        """
        firstAdapter = lambda o: False
        secondAdapter = lambda o: True

        class TheInterface(Interface):
            pass
        components.registerAdapter(firstAdapter, original, TheInterface)
        components.ALLOW_DUPLICATES = True
        try:
            components.registerAdapter(secondAdapter, original, TheInterface)
            self.assertIs(components.getAdapterFactory(original, TheInterface, None), secondAdapter)
        finally:
            components.ALLOW_DUPLICATES = False
        self.assertRaises(ValueError, components.registerAdapter, firstAdapter, original, TheInterface)
        self.assertIs(components.getAdapterFactory(original, TheInterface, None), secondAdapter)

    def test_duplicateAdapterForClassAllowed(self):
        """
        Test that when L{components.ALLOW_DUPLICATES} is set to a true
        value, duplicate registrations from classes are allowed to override
        the original registration.
        """

        class TheOriginal:
            pass
        return self._duplicateAdapterForClassOrInterfaceAllowed(TheOriginal)

    def test_duplicateAdapterForInterfaceAllowed(self):
        """
        Test that when L{components.ALLOW_DUPLICATES} is set to a true
        value, duplicate registrations from interfaces are allowed to
        override the original registration.
        """

        class TheOriginal(Interface):
            pass
        return self._duplicateAdapterForClassOrInterfaceAllowed(TheOriginal)

    def _multipleInterfacesForClassOrInterface(self, original):
        """
        Verify that an adapter can be registered for multiple to-interfaces at a
        time.
        """
        adapter = lambda o: None
        components.registerAdapter(adapter, original, ITest, ITest2)
        self.assertIs(components.getAdapterFactory(original, ITest, None), adapter)
        self.assertIs(components.getAdapterFactory(original, ITest2, None), adapter)

    def test_multipleInterfacesForClass(self):
        """
        Test the registration of an adapter from a class to several
        interfaces at once.
        """

        class TheOriginal:
            pass
        return self._multipleInterfacesForClassOrInterface(TheOriginal)

    def test_multipleInterfacesForInterface(self):
        """
        Test the registration of an adapter from an interface to several
        interfaces at once.
        """
        return self._multipleInterfacesForClassOrInterface(ITest3)

    def _subclassAdapterRegistrationForClassOrInterface(self, original):
        """
        Verify that a new adapter can be registered for a particular
        to-interface from a subclass of a type or interface which already has an
        adapter registered to that interface and that the subclass adapter takes
        precedence over the base class adapter.
        """
        firstAdapter = lambda o: True
        secondAdapter = lambda o: False

        class TheSubclass(original):
            pass
        components.registerAdapter(firstAdapter, original, ITest)
        components.registerAdapter(secondAdapter, TheSubclass, ITest)
        self.assertIs(components.getAdapterFactory(original, ITest, None), firstAdapter)
        self.assertIs(components.getAdapterFactory(TheSubclass, ITest, None), secondAdapter)

    def test_subclassAdapterRegistrationForClass(self):
        """
        Test that an adapter to a particular interface can be registered
        from both a class and its subclass.
        """

        class TheOriginal:
            pass
        return self._subclassAdapterRegistrationForClassOrInterface(TheOriginal)

    def test_subclassAdapterRegistrationForInterface(self):
        """
        Test that an adapter to a particular interface can be registered
        from both an interface and its subclass.
        """
        return self._subclassAdapterRegistrationForClassOrInterface(ITest2)