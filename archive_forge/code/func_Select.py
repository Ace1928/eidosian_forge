from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import time
from googlecloudsdk.core.cache import exceptions
import six
import six.moves.urllib.parse
@abc.abstractmethod
def Select(self, name=None):
    """Returns the list of table names matching name.

    Args:
      name: The table name pattern to match, None for all tables. The pattern
        may contain these wildcard characters:
          * - match zero or more characters
          ? - match any character
    """
    pass