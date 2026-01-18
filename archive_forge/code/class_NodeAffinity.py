from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeAffinity(_messages.Message):
    """Node Affinity: the configuration of desired nodes onto which this
  Instance could be scheduled.

  Enums:
    OperatorValueValuesEnum: Optional. Defines the operation of node
      selection.

  Fields:
    key: Optional. Corresponds to the label key of Node resource.
    operator: Optional. Defines the operation of node selection.
    values: Optional. Corresponds to the label values of Node resource.
  """

    class OperatorValueValuesEnum(_messages.Enum):
        """Optional. Defines the operation of node selection.

    Values:
      OPERATOR_UNSPECIFIED: Default value. This value is unused.
      IN: Requires Compute Engine to seek for matched nodes.
      NOT_IN: Requires Compute Engine to avoid certain nodes.
    """
        OPERATOR_UNSPECIFIED = 0
        IN = 1
        NOT_IN = 2
    key = _messages.StringField(1)
    operator = _messages.EnumField('OperatorValueValuesEnum', 2)
    values = _messages.StringField(3, repeated=True)