from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from absl.flags import _helpers
class FlagNameConflictsWithMethodError(Error):
    """Raised when a flag name conflicts with FlagValues methods."""