import warnings
from twisted.trial.unittest import TestCase
class NamesTests(TestCase, _ConstantsTestsMixin):
    """
    Tests for L{twisted.python.constants.Names}, a base class for containers of
    related constraints.
    """

    def setUp(self):
        """
        Create a fresh new L{Names} subclass for each unit test to use.  Since
        L{Names} is stateful, re-using the same subclass across test methods
        makes exercising all of the implementation code paths difficult.
        """

        class METHOD(Names):
            """
            A container for some named constants to use in unit tests for
            L{Names}.
            """
            GET = NamedConstant()
            PUT = NamedConstant()
            POST = NamedConstant()
            DELETE = NamedConstant()
            extra = object()
        self.METHOD = METHOD

    def test_notInstantiable(self):
        """
        A subclass of L{Names} raises C{TypeError} if an attempt is made to
        instantiate it.
        """
        self._notInstantiableTest('METHOD', self.METHOD)

    def test_symbolicAttributes(self):
        """
        Each name associated with a L{NamedConstant} instance in the definition
        of a L{Names} subclass is available as an attribute on the resulting
        class.
        """
        self.assertTrue(hasattr(self.METHOD, 'GET'))
        self.assertTrue(hasattr(self.METHOD, 'PUT'))
        self.assertTrue(hasattr(self.METHOD, 'POST'))
        self.assertTrue(hasattr(self.METHOD, 'DELETE'))

    def test_withoutOtherAttributes(self):
        """
        As usual, names not defined in the class scope of a L{Names}
        subclass are not available as attributes on the resulting class.
        """
        self.assertFalse(hasattr(self.METHOD, 'foo'))

    def test_representation(self):
        """
        The string representation of a constant on a L{Names} subclass includes
        the name of the L{Names} subclass and the name of the constant itself.
        """
        self.assertEqual('<METHOD=GET>', repr(self.METHOD.GET))

    def test_lookupByName(self):
        """
        Constants can be looked up by name using L{Names.lookupByName}.
        """
        method = self.METHOD.lookupByName('GET')
        self.assertIs(self.METHOD.GET, method)

    def test_notLookupMissingByName(self):
        """
        Names not defined with a L{NamedConstant} instance cannot be looked up
        using L{Names.lookupByName}.
        """
        self.assertRaises(ValueError, self.METHOD.lookupByName, 'lookupByName')
        self.assertRaises(ValueError, self.METHOD.lookupByName, '__init__')
        self.assertRaises(ValueError, self.METHOD.lookupByName, 'foo')
        self.assertRaises(ValueError, self.METHOD.lookupByName, 'extra')

    def test_name(self):
        """
        The C{name} attribute of one of the named constants gives that
        constant's name.
        """
        self.assertEqual('GET', self.METHOD.GET.name)

    def test_attributeIdentity(self):
        """
        Repeated access of an attribute associated with a L{NamedConstant}
        value in a L{Names} subclass results in the same object.
        """
        self.assertIs(self.METHOD.GET, self.METHOD.GET)

    def test_iterconstants(self):
        """
        L{Names.iterconstants} returns an iterator over all of the constants
        defined in the class, in the order they were defined.
        """
        constants = list(self.METHOD.iterconstants())
        self.assertEqual([self.METHOD.GET, self.METHOD.PUT, self.METHOD.POST, self.METHOD.DELETE], constants)

    def test_attributeIterconstantsIdentity(self):
        """
        The constants returned from L{Names.iterconstants} are identical to the
        constants accessible using attributes.
        """
        constants = list(self.METHOD.iterconstants())
        self.assertIs(self.METHOD.GET, constants[0])
        self.assertIs(self.METHOD.PUT, constants[1])
        self.assertIs(self.METHOD.POST, constants[2])
        self.assertIs(self.METHOD.DELETE, constants[3])

    def test_iterconstantsIdentity(self):
        """
        The constants returned from L{Names.iterconstants} are identical on
        each call to that method.
        """
        constants = list(self.METHOD.iterconstants())
        again = list(self.METHOD.iterconstants())
        self.assertIs(again[0], constants[0])
        self.assertIs(again[1], constants[1])
        self.assertIs(again[2], constants[2])
        self.assertIs(again[3], constants[3])

    def test_initializedOnce(self):
        """
        L{Names._enumerants} is initialized once and its value re-used on
        subsequent access.
        """
        self._initializedOnceTest(self.METHOD, 'GET')

    def test_asForeignClassAttribute(self):
        """
        A constant defined on a L{Names} subclass may be set as an attribute of
        another class and then retrieved using that attribute.
        """

        class Another:
            something = self.METHOD.GET
        self.assertIs(self.METHOD.GET, Another.something)

    def test_asForeignClassAttributeViaInstance(self):
        """
        A constant defined on a L{Names} subclass may be set as an attribute of
        another class and then retrieved from an instance of that class using
        that attribute.
        """

        class Another:
            something = self.METHOD.GET
        self.assertIs(self.METHOD.GET, Another().something)

    def test_notAsAlternateContainerAttribute(self):
        """
        It is explicitly disallowed (via a L{ValueError}) to use a constant
        defined on a L{Names} subclass as the value of an attribute of another
        L{Names} subclass.
        """

        def defineIt():

            class AnotherNames(Names):
                something = self.METHOD.GET
        exc = self.assertRaises(ValueError, defineIt)
        self.assertEqual('Cannot use <METHOD=GET> as the value of an attribute on AnotherNames', str(exc))