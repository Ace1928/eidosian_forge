from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchedulingNodeAffinity(_messages.Message):
    """Node Affinity: the configuration of desired nodes onto which this
  Instance could be scheduled. Based on https://cloud.google.com/compute/docs/
  reference/rest/v1/instances/setScheduling

  Enums:
    OperatorValueValuesEnum: The operator to use for the node resources
      specified in the `values` parameter.

  Fields:
    key: The label key of Node resource to reference.
    operator: The operator to use for the node resources specified in the
      `values` parameter.
    values: Corresponds to the label values of Node resource.
  """

    class OperatorValueValuesEnum(_messages.Enum):
        """The operator to use for the node resources specified in the `values`
    parameter.

    Values:
      OPERATOR_UNSPECIFIED: An unknown, unexpected behavior.
      IN: The node resource group should be in these resources affinity.
      NOT_IN: The node resource group should not be in these resources
        affinity.
    """
        OPERATOR_UNSPECIFIED = 0
        IN = 1
        NOT_IN = 2
    key = _messages.StringField(1)
    operator = _messages.EnumField('OperatorValueValuesEnum', 2)
    values = _messages.StringField(3, repeated=True)