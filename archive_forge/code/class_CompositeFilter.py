from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CompositeFilter(_messages.Message):
    """A filter that merges multiple other filters using the given operator.

  Enums:
    OpValueValuesEnum: The operator for combining multiple filters.

  Fields:
    filters: The list of filters to combine. Requires: * At least one filter
      is present.
    op: The operator for combining multiple filters.
  """

    class OpValueValuesEnum(_messages.Enum):
        """The operator for combining multiple filters.

    Values:
      OPERATOR_UNSPECIFIED: Unspecified. This value must not be used.
      AND: Documents are required to satisfy all of the combined filters.
      OR: Documents are required to satisfy at least one of the combined
        filters.
    """
        OPERATOR_UNSPECIFIED = 0
        AND = 1
        OR = 2
    filters = _messages.MessageField('Filter', 1, repeated=True)
    op = _messages.EnumField('OpValueValuesEnum', 2)