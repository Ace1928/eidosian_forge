from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import sys
def DelAttribute(self, name):
    """Deletes name from the attributes if it is in the attributes.

    Args:
      name: The attribute name.
    """
    if name in self.attributes:
        del self.attributes[name]