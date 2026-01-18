from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CustomResourceColumnDefinition(_messages.Message):
    """CustomResourceColumnDefinition specifies a column for server side
  printing.

  Fields:
    description: description is a human readable description of this column.
    format: format is an optional OpenAPI type definition for this column. The
      'name' format is applied to the primary identifier column to assist in
      clients identifying column is the resource name. See
      https://github.com/OAI/OpenAPI-
      Specification/blob/master/versions/2.0.md#data-types for more.
    jsonPath: JSONPath is a simple JSON path, i.e. with array notation.
    name: name is a human readable name for the column.
    priority: priority is an integer defining the relative importance of this
      column compared to others. Lower numbers are considered higher priority.
      Columns that may be omitted in limited space scenarios should be given a
      higher priority. +optional
    type: type is an OpenAPI type definition for this column. See
      https://github.com/OAI/OpenAPI-
      Specification/blob/master/versions/2.0.md#data-types for more.
  """
    description = _messages.StringField(1)
    format = _messages.StringField(2)
    jsonPath = _messages.StringField(3)
    name = _messages.StringField(4)
    priority = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    type = _messages.StringField(6)