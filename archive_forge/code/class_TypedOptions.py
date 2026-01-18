from twisted.python import usage
from twisted.trial import unittest
class TypedOptions(usage.Options):
    optParameters = [['fooint', None, 392, 'Foo int', int], ['foofloat', None, 4.23, 'Foo float', float], ['eggint', None, None, 'Egg int without default', int], ['eggfloat', None, None, 'Egg float without default', float]]

    def opt_under_score(self, value):
        """
        This option has an underscore in its name to exercise the _ to -
        translation.
        """
        self.underscoreValue = value
    opt_u = opt_under_score