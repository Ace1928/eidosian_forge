from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class CsObjectOrPrefix(object):
    """Container class for ListObjects results."""

    def __init__(self, data, datatype):
        """Stores a ListObjects result.

      Args:
        data: Root object, either an apitools Object or a string Prefix.
        datatype: CsObjectOrPrefixType of data.
      """
        self.data = data
        self.datatype = datatype