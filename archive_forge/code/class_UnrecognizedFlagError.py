from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from absl.flags import _helpers
class UnrecognizedFlagError(Error):
    """Raised when a flag is unrecognized.

  Attributes:
    flagname: str, the name of the unrecognized flag.
    flagvalue: The value of the flag, empty if the flag is not defined.
  """

    def __init__(self, flagname, flagvalue='', suggestions=None):
        self.flagname = flagname
        self.flagvalue = flagvalue
        if suggestions:
            tip = '. Did you mean: %s ?' % ', '.join(suggestions)
        else:
            tip = ''
        super(UnrecognizedFlagError, self).__init__("Unknown command line flag '%s'%s" % (flagname, tip))