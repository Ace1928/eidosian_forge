import sys, os
import textwrap
class AmbiguousOptionError(BadOptionError):
    """
    Raised if an ambiguous option is seen on the command line.
    """

    def __init__(self, opt_str, possibilities):
        BadOptionError.__init__(self, opt_str)
        self.possibilities = possibilities

    def __str__(self):
        return _('ambiguous option: %s (%s?)') % (self.opt_str, ', '.join(self.possibilities))