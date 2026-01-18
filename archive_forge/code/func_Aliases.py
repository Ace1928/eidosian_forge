from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import sys
def Aliases(self):
    """Returns the short key name alias dictionary.

    This dictionary maps short (no dots) names to parsed keys.

    Returns:
      The short key name alias dictionary.
    """
    return self.aliases