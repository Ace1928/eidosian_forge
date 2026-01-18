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
class _TableColumn(object):
    """A wrapper that stores the column information.

  Attributes:
    name: String, the name of the table column.
    col_type: _ScalarColumnType or _ArrayColumnType.
  """
    _COLUMN_DDL_PATTERN = re.compile('\n            # A column definition has a name and a type, with some additional\n            # properties.\n            # Some examples:\n            #    Foo INT64 NOT NULL\n            #    Bar STRING(1024)\n            #    Baz ARRAY<FLOAT32>\n            [`]?(?P<name>\\w+)[`]?\\s+\n            (?P<type>[\\w<>]+)\n            # We don\'t care about "NOT NULL", and the length number after STRING\n            # or BYTES (e.g.STRING(MAX), BYTES(1024)).\n        ', re.DOTALL | re.VERBOSE)

    def __init__(self, name, col_type):
        self.name = name
        self.col_type = col_type

    def __eq__(self, other):
        return self.name == other.name and self.col_type == other.col_type

    @classmethod
    def FromDdl(cls, column_ddl):
        """Constructs an instance of _TableColumn from a column_def DDL statement.

    Args:
      column_ddl: string, the parsed string contains the column name and type
        information. Example: SingerId INT64 NOT NULL.

    Returns:
      A _TableColumn object.

    Raises:
      ValueError: invalid DDL, this error shouldn't happen in theory, as the API
        is expected to return valid DDL statement strings.
    """
        column_match = cls._COLUMN_DDL_PATTERN.search(column_ddl)
        if not column_match:
            raise ValueError('Invalid DDL: [{}].'.format(column_ddl))
        column_name = column_match.group('name')
        col_type = _ColumnType.FromDdl(column_match.group('type'))
        return _TableColumn(column_name, col_type)

    def GetJsonValues(self, value):
        """Convert the user input values to JSON value or JSON array value.

    Args:
      value: String or string list, the user input values of the column.

    Returns:
      extra_types.JsonArray or extra_types.JsonValue, the json value of a single
          column in the format that API accepts.
    """
        return self.col_type.GetJsonValue(value)