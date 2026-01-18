from twisted.python import usage
from twisted.trial import unittest
class Opts(usage.Options):

    def opt_very_very_long(self):
        """
                This is an option method with a very long name, that is going to
                be aliased.
                """
    opt_short = opt_very_very_long
    opt_s = opt_very_very_long