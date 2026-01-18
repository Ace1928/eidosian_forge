from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceQuotaLifecycleState(_messages.Message):
    """ResourceQuotaLifecycleState represents lifecycle state for
  ResourceQuota.

  Enums:
    CodeValueValuesEnum: Output only. The current state of the ResourceQuota
      resource.

  Fields:
    code: Output only. The current state of the ResourceQuota resource.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Output only. The current state of the ResourceQuota resource.

    Values:
      CODE_UNSPECIFIED: The code is not set.
      CREATING: The resourcequota is being created.
      READY: The resourcequota active.
      DELETING: The resourcequota is being deleted.
      UPDATING: The resourcequota is being updated.
    """
        CODE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
        DELETING = 3
        UPDATING = 4
    code = _messages.EnumField('CodeValueValuesEnum', 1)