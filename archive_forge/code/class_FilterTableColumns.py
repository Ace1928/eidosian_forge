from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FilterTableColumns(_messages.Message):
    """Options to configure rule type FilterTableColumns. The rule is used to
  filter the list of columns to include or exclude from a table. The rule
  filter field can refer to one entity. The rule scope can be: Table Only one
  of the two lists can be specified for the rule.

  Fields:
    excludeColumns: Optional. List of columns to be excluded for a particular
      table.
    includeColumns: Optional. List of columns to be included for a particular
      table.
  """
    excludeColumns = _messages.StringField(1, repeated=True)
    includeColumns = _messages.StringField(2, repeated=True)