import warnings
from twisted.trial.unittest import TestCase
class NamedConstantTests(TestCase):
    """
    Tests for the L{twisted.python.constants.NamedConstant} class which is used
    to represent individual values.
    """

    def setUp(self):
        """
        Create a dummy container into which constants can be placed.
        """

        class foo(Names):
            pass
        self.container = foo

    def test_name(self):
        """
        The C{name} attribute of a L{NamedConstant} refers to the value passed
        for the C{name} parameter to C{_realize}.
        """
        name = NamedConstant()
        name._realize(self.container, 'bar', None)
        self.assertEqual('bar', name.name)

    def test_representation(self):
        """
        The string representation of an instance of L{NamedConstant} includes
        the container the instances belongs to as well as the instance's name.
        """
        name = NamedConstant()
        name._realize(self.container, 'bar', None)
        self.assertEqual('<foo=bar>', repr(name))

    def test_equality(self):
        """
        A L{NamedConstant} instance compares equal to itself.
        """
        name = NamedConstant()
        name._realize(self.container, 'bar', None)
        self.assertTrue(name == name)
        self.assertFalse(name != name)

    def test_nonequality(self):
        """
        Two different L{NamedConstant} instances do not compare equal to each
        other.
        """
        first = NamedConstant()
        first._realize(self.container, 'bar', None)
        second = NamedConstant()
        second._realize(self.container, 'bar', None)
        self.assertFalse(first == second)
        self.assertTrue(first != second)

    def test_hash(self):
        """
        Because two different L{NamedConstant} instances do not compare as
        equal to each other, they also have different hashes to avoid
        collisions when added to a C{dict} or C{set}.
        """
        first = NamedConstant()
        first._realize(self.container, 'bar', None)
        second = NamedConstant()
        second._realize(self.container, 'bar', None)
        self.assertNotEqual(hash(first), hash(second))