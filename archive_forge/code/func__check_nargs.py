import sys, os
import textwrap
def _check_nargs(self):
    if self.action in self.TYPED_ACTIONS:
        if self.nargs is None:
            self.nargs = 1
    elif self.nargs is not None:
        raise OptionError("'nargs' must not be supplied for action %r" % self.action, self)