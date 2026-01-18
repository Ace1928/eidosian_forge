from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadIdentityPoolNamespace(_messages.Message):
    """Represents a namespace for a workload identity pool. Namespaces are used
  to segment identities within the pool.

  Enums:
    StateValueValuesEnum: Output only. The state of the namespace.

  Fields:
    description: A description of the namespace. Cannot exceed 256 characters.
    disabled: Whether the namespace is disabled. If disabled, credentials may
      no longer be issued for identities within this namespace, however
      existing credentials will still be accepted until they expire.
    expireTime: Output only. Time after which the namespace will be
      permanently purged and cannot be recovered.
    name: Output only. The resource name of the namespace.
    state: Output only. The state of the namespace.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the namespace.

    Values:
      STATE_UNSPECIFIED: State unspecified.
      ACTIVE: The namespace is active.
      DELETED: The namespace is soft-deleted. Soft-deleted namespaces are
        permanently deleted after approximately 30 days. You can restore a
        soft-deleted namespace using UndeleteWorkloadIdentityPoolNamespace.
        You cannot reuse the ID of a soft-deleted namespace until it is
        permanently deleted.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DELETED = 2
    description = _messages.StringField(1)
    disabled = _messages.BooleanField(2)
    expireTime = _messages.StringField(3)
    name = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)