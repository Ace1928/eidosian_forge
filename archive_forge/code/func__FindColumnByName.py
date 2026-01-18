from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from collections import OrderedDict
import re
from apitools.base.py import extra_types
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import zip
def _FindColumnByName(self, col_name):
    """Find the _TableColumn object with the given column name.

    Args:
      col_name: String, the name of the column.

    Returns:
      _TableColumn.

    Raises:
      BadColumnNameError: the column name is invalid.
    """
    try:
        return self._columns[col_name]
    except KeyError:
        valid_column_names = ', '.join(list(self._columns.keys()))
        raise BadColumnNameError('Column name [{}] is invalid. Valid column names: [{}].'.format(col_name, valid_column_names))