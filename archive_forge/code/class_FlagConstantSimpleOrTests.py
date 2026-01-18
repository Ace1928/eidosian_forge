import warnings
from twisted.trial.unittest import TestCase
class FlagConstantSimpleOrTests(_FlagsTestsMixin, TestCase):
    """
    Tests for the C{|} operator as defined for L{FlagConstant} instances, used
    to create new L{FlagConstant} instances representing both of two existing
    L{FlagConstant} instances from the same L{Flags} class.
    """

    def test_value(self):
        """
        The value of the L{FlagConstant} which results from C{|} has all of the
        bits set which were set in either of the values of the two original
        constants.
        """
        flag = self.FXF.READ | self.FXF.WRITE
        self.assertEqual(self.FXF.READ.value | self.FXF.WRITE.value, flag.value)

    def test_name(self):
        """
        The name of the L{FlagConstant} instance which results from C{|}
        includes the names of both of the two original constants.
        """
        flag = self.FXF.READ | self.FXF.WRITE
        self.assertEqual('{READ,WRITE}', flag.name)

    def test_representation(self):
        """
        The string representation of a L{FlagConstant} instance which results
        from C{|} includes the names of both of the two original constants.
        """
        flag = self.FXF.READ | self.FXF.WRITE
        self.assertEqual('<FXF={READ,WRITE}>', repr(flag))

    def test_iterate(self):
        """
        A L{FlagConstant} instance which results from C{|} can be
        iterated upon to yield the original constants.
        """
        self.assertEqual(set(self.FXF.WRITE & self.FXF.READ), set())
        self.assertEqual(set(self.FXF.WRITE), {self.FXF.WRITE})
        self.assertEqual(set(self.FXF.WRITE | self.FXF.EXCLUSIVE), {self.FXF.WRITE, self.FXF.EXCLUSIVE})

    def test_membership(self):
        """
        A L{FlagConstant} instance which results from C{|} can be
        tested for membership.
        """
        flags = self.FXF.WRITE | self.FXF.EXCLUSIVE
        self.assertIn(self.FXF.WRITE, flags)
        self.assertNotIn(self.FXF.READ, flags)

    def test_truthiness(self):
        """
        Empty flags is false, non-empty flags is true.
        """
        self.assertTrue(self.FXF.WRITE)
        self.assertTrue(self.FXF.WRITE | self.FXF.EXCLUSIVE)
        self.assertFalse(self.FXF.WRITE & self.FXF.EXCLUSIVE)