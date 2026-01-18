from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PlacementPolicy(_messages.Message):
    """PlacementPolicy defines the placement policy used by the node pool.

  Enums:
    TypeValueValuesEnum: The type of placement.

  Fields:
    policyName: If set, refers to the name of a custom resource policy
      supplied by the user. The resource policy must be in the same project
      and region as the node pool. If not found, InvalidArgument error is
      returned.
    tpuTopology: Optional. TPU placement topology for pod slice node pool.
      https://cloud.google.com/tpu/docs/types-topologies#tpu_topologies
    type: The type of placement.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of placement.

    Values:
      TYPE_UNSPECIFIED: TYPE_UNSPECIFIED specifies no requirements on nodes
        placement.
      COMPACT: COMPACT specifies node placement in the same availability
        domain to ensure low communication latency.
    """
        TYPE_UNSPECIFIED = 0
        COMPACT = 1
    policyName = _messages.StringField(1)
    tpuTopology = _messages.StringField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)