from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import time
from googlecloudsdk.core.cache import exceptions
import six
import six.moves.urllib.parse
def _CheckRowTemplates(self, rows):
    """Raise an exception if the size of any row template in rows is invalid.

    Each row template must have at least 1 column and no more than the number
    of columns in the table.

    Args:
      rows: The list of rows to check.

    Raises:
      CacheTableRowSizeInvalid: If any row template size is invalid.
    """
    for row in rows:
        if not 1 <= len(row) <= self.columns:
            if self.columns == 1:
                limits = '1'
            else:
                limits = '>= 1 and <= {}'.format(self.columns)
            raise exceptions.CacheTableRowSizeInvalid('Cache table [{}] row size [{}] is invalid. Must be {}.'.format(self.name, len(row), limits))