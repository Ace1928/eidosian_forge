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
class _ColumnType(six.with_metaclass(abc.ABCMeta, object)):
    """A wrapper that stores the column type information.

  A column type can be one of the scalar types such as integers, as well as
      array. An array type is an ordered list of zero or more elements of
      scalar type.

  Attributes:
    scalar_type: String, the type of the column or the element of the column
        (if the column is an array).
  """
    _SCALAR_TYPES = ('BOOL', 'BYTES', 'DATE', 'FLOAT64', 'INT64', 'STRING', 'TIMESTAMP', 'NUMERIC', 'JSON', 'TOKENLIST')

    def __init__(self, scalar_type):
        self.scalar_type = scalar_type

    @classmethod
    def FromDdl(cls, column_type_ddl):
        """Constructs a _ColumnType object from a partial DDL statement.

    Args:
      column_type_ddl: string, the parsed string only contains the column type
        information. Example: INT64 NOT NULL, ARRAY<STRING(MAX)> or BYTES(200).

    Returns:
      A _ArrayColumnType or a _ScalarColumnType object.

    Raises:
      ValueError: invalid DDL, this error shouldn't happen in theory, as the API
        is expected to return valid DDL statement strings.
    """
        scalar_match = None
        for data_type in cls._SCALAR_TYPES:
            if data_type in column_type_ddl:
                scalar_match = data_type
                break
        if not scalar_match:
            raise ValueError('Invalid DDL: unrecognized type [{}].'.format(column_type_ddl))
        if column_type_ddl.startswith('ARRAY'):
            return _ArrayColumnType(scalar_match)
        else:
            return _ScalarColumnType(scalar_match)

    @abc.abstractmethod
    def GetJsonValue(self, value):
        raise NotImplementedError()