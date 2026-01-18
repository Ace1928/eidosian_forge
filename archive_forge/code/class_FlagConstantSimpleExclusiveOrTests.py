import warnings
from twisted.trial.unittest import TestCase
class FlagConstantSimpleExclusiveOrTests(_FlagsTestsMixin, TestCase):
    """
    Tests for the C{^} operator as defined for L{FlagConstant} instances, used
    to create new L{FlagConstant} instances representing the uncommon parts of
    two existing L{FlagConstant} instances from the same L{Flags} class.
    """

    def test_value(self):
        """
        The value of the L{FlagConstant} which results from C{^} has all of the
        bits set which were set in exactly one of the values of the two
        original constants.
        """
        readWrite = self.FXF.READ | self.FXF.WRITE
        writeAppend = self.FXF.WRITE | self.FXF.APPEND
        flag = readWrite ^ writeAppend
        self.assertEqual(self.FXF.READ.value | self.FXF.APPEND.value, flag.value)

    def test_name(self):
        """
        The name of the L{FlagConstant} instance which results from C{^}
        includes the names of only the flags which were set in exactly one of
        the two original constants.
        """
        readWrite = self.FXF.READ | self.FXF.WRITE
        writeAppend = self.FXF.WRITE | self.FXF.APPEND
        flag = readWrite ^ writeAppend
        self.assertEqual('{APPEND,READ}', flag.name)

    def test_representation(self):
        """
        The string representation of a L{FlagConstant} instance which results
        from C{^} includes the names of only the flags which were set in
        exactly one of the two original constants.
        """
        readWrite = self.FXF.READ | self.FXF.WRITE
        writeAppend = self.FXF.WRITE | self.FXF.APPEND
        flag = readWrite ^ writeAppend
        self.assertEqual('<FXF={APPEND,READ}>', repr(flag))