from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import sys
def AddAttribute(self, name, value):
    """Adds name=value to the attributes.

    Args:
      name: The attribute name.
      value: The attribute value
    """
    self.attributes[name] = value