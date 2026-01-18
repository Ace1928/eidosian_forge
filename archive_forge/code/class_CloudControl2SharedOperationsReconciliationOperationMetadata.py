from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudControl2SharedOperationsReconciliationOperationMetadata(_messages.Message):
    """Operation metadata returned by the CLH during resource state
  reconciliation.

  Enums:
    ExclusiveActionValueValuesEnum: Excluisive action returned by the CLH.

  Fields:
    deleteResource: DEPRECATED. Use exclusive_action instead.
    exclusiveAction: Excluisive action returned by the CLH.
  """

    class ExclusiveActionValueValuesEnum(_messages.Enum):
        """Excluisive action returned by the CLH.

    Values:
      UNKNOWN_REPAIR_ACTION: Unknown repair action.
      DELETE: The resource has to be deleted. When using this bit, the CLH
        should fail the operation. DEPRECATED. Instead use DELETE_RESOURCE
        OperationSignal in SideChannel.
      RETRY: This resource could not be repaired but the repair should be
        tried again at a later time. This can happen if there is a dependency
        that needs to be resolved first- e.g. if a parent resource must be
        repaired before a child resource.
    """
        UNKNOWN_REPAIR_ACTION = 0
        DELETE = 1
        RETRY = 2
    deleteResource = _messages.BooleanField(1)
    exclusiveAction = _messages.EnumField('ExclusiveActionValueValuesEnum', 2)