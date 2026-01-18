import warnings
from twisted.trial.unittest import TestCase
class ValuesTests(TestCase, _ConstantsTestsMixin):
    """
    Tests for L{twisted.python.constants.Names}, a base class for containers of
    related constraints with arbitrary values.
    """

    def setUp(self):
        """
        Create a fresh new L{Values} subclass for each unit test to use.  Since
        L{Values} is stateful, re-using the same subclass across test methods
        makes exercising all of the implementation code paths difficult.
        """

        class STATUS(Values):
            OK = ValueConstant('200')
            NOT_FOUND = ValueConstant('404')
        self.STATUS = STATUS

    def test_notInstantiable(self):
        """
        A subclass of L{Values} raises C{TypeError} if an attempt is made to
        instantiate it.
        """
        self._notInstantiableTest('STATUS', self.STATUS)

    def test_symbolicAttributes(self):
        """
        Each name associated with a L{ValueConstant} instance in the definition
        of a L{Values} subclass is available as an attribute on the resulting
        class.
        """
        self.assertTrue(hasattr(self.STATUS, 'OK'))
        self.assertTrue(hasattr(self.STATUS, 'NOT_FOUND'))

    def test_withoutOtherAttributes(self):
        """
        As usual, names not defined in the class scope of a L{Values}
        subclass are not available as attributes on the resulting class.
        """
        self.assertFalse(hasattr(self.STATUS, 'foo'))

    def test_representation(self):
        """
        The string representation of a constant on a L{Values} subclass
        includes the name of the L{Values} subclass and the name of the
        constant itself.
        """
        self.assertEqual('<STATUS=OK>', repr(self.STATUS.OK))

    def test_lookupByName(self):
        """
        Constants can be looked up by name using L{Values.lookupByName}.
        """
        method = self.STATUS.lookupByName('OK')
        self.assertIs(self.STATUS.OK, method)

    def test_notLookupMissingByName(self):
        """
        Names not defined with a L{ValueConstant} instance cannot be looked up
        using L{Values.lookupByName}.
        """
        self.assertRaises(ValueError, self.STATUS.lookupByName, 'lookupByName')
        self.assertRaises(ValueError, self.STATUS.lookupByName, '__init__')
        self.assertRaises(ValueError, self.STATUS.lookupByName, 'foo')

    def test_lookupByValue(self):
        """
        Constants can be looked up by their associated value, defined by the
        argument passed to L{ValueConstant}, using L{Values.lookupByValue}.
        """
        status = self.STATUS.lookupByValue('200')
        self.assertIs(self.STATUS.OK, status)

    def test_lookupDuplicateByValue(self):
        """
        If more than one constant is associated with a particular value,
        L{Values.lookupByValue} returns whichever of them is defined first.
        """

        class TRANSPORT_MESSAGE(Values):
            """
            Message types supported by an SSH transport.
            """
            KEX_DH_GEX_REQUEST_OLD = ValueConstant(30)
            KEXDH_INIT = ValueConstant(30)
        self.assertIs(TRANSPORT_MESSAGE.lookupByValue(30), TRANSPORT_MESSAGE.KEX_DH_GEX_REQUEST_OLD)

    def test_notLookupMissingByValue(self):
        """
        L{Values.lookupByValue} raises L{ValueError} when called with a value
        with which no constant is associated.
        """
        self.assertRaises(ValueError, self.STATUS.lookupByValue, 'OK')
        self.assertRaises(ValueError, self.STATUS.lookupByValue, 200)
        self.assertRaises(ValueError, self.STATUS.lookupByValue, '200.1')

    def test_name(self):
        """
        The C{name} attribute of one of the constants gives that constant's
        name.
        """
        self.assertEqual('OK', self.STATUS.OK.name)

    def test_attributeIdentity(self):
        """
        Repeated access of an attribute associated with a L{ValueConstant}
        value in a L{Values} subclass results in the same object.
        """
        self.assertIs(self.STATUS.OK, self.STATUS.OK)

    def test_iterconstants(self):
        """
        L{Values.iterconstants} returns an iterator over all of the constants
        defined in the class, in the order they were defined.
        """
        constants = list(self.STATUS.iterconstants())
        self.assertEqual([self.STATUS.OK, self.STATUS.NOT_FOUND], constants)

    def test_attributeIterconstantsIdentity(self):
        """
        The constants returned from L{Values.iterconstants} are identical to
        the constants accessible using attributes.
        """
        constants = list(self.STATUS.iterconstants())
        self.assertIs(self.STATUS.OK, constants[0])
        self.assertIs(self.STATUS.NOT_FOUND, constants[1])

    def test_iterconstantsIdentity(self):
        """
        The constants returned from L{Values.iterconstants} are identical on
        each call to that method.
        """
        constants = list(self.STATUS.iterconstants())
        again = list(self.STATUS.iterconstants())
        self.assertIs(again[0], constants[0])
        self.assertIs(again[1], constants[1])

    def test_initializedOnce(self):
        """
        L{Values._enumerants} is initialized once and its value re-used on
        subsequent access.
        """
        self._initializedOnceTest(self.STATUS, 'OK')