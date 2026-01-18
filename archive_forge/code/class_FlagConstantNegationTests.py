import warnings
from twisted.trial.unittest import TestCase
class FlagConstantNegationTests(_FlagsTestsMixin, TestCase):
    """
    Tests for the C{~} operator as defined for L{FlagConstant} instances, used
    to create new L{FlagConstant} instances representing all the flags from a
    L{Flags} class not set in a particular L{FlagConstant} instance.
    """

    def test_value(self):
        """
        The value of the L{FlagConstant} which results from C{~} has all of the
        bits set which were not set in the original constant.
        """
        flag = ~self.FXF.READ
        self.assertEqual(self.FXF.WRITE.value | self.FXF.APPEND.value | self.FXF.EXCLUSIVE.value | self.FXF.TEXT.value, flag.value)
        flag = ~self.FXF.WRITE
        self.assertEqual(self.FXF.READ.value | self.FXF.APPEND.value | self.FXF.EXCLUSIVE.value | self.FXF.TEXT.value, flag.value)

    def test_name(self):
        """
        The name of the L{FlagConstant} instance which results from C{~}
        includes the names of all the flags which were not set in the original
        constant.
        """
        flag = ~self.FXF.WRITE
        self.assertEqual('{APPEND,EXCLUSIVE,READ,TEXT}', flag.name)

    def test_representation(self):
        """
        The string representation of a L{FlagConstant} instance which results
        from C{~} includes the names of all the flags which were not set in the
        original constant.
        """
        flag = ~self.FXF.WRITE
        self.assertEqual('<FXF={APPEND,EXCLUSIVE,READ,TEXT}>', repr(flag))