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
def ValidateArrayInput(table, data):
    """Checks array input is valid.

  Args:
    table: Table, the table which data is being modified.
    data: OrderedDict, the data entered by the user.

  Returns:
    data (OrderedDict) the validated data.

  Raises:
    InvalidArrayInputError: if the input contains an array which is invalid.
  """
    col_to_type = table.GetColumnTypes()
    for column, value in six.iteritems(data):
        col_type = col_to_type[column]
        if isinstance(col_type, _ArrayColumnType) and isinstance(value, list) is False:
            raise InvalidArrayInputError('Column name [{}] has an invalid array input: {}. `--flags-file` should be used to specify array values.'.format(column, value))
    return data