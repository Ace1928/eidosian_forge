from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1TtlConfig(_messages.Message):
    """The TTL (time-to-live) configuration for documents that have this
  `Field` set. Storing a timestamp value into a TTL-enabled field will be
  treated as the document's absolute expiration time. Timestamp values in the
  past indicate that the document is eligible for immediate expiration. Using
  any other data type or leaving the field absent will disable expiration for
  the individual document.

  Enums:
    StateValueValuesEnum: Output only. The state of the TTL configuration.

  Fields:
    state: Output only. The state of the TTL configuration.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the TTL configuration.

    Values:
      STATE_UNSPECIFIED: The state is unspecified or unknown.
      CREATING: The TTL is being applied. There is an active long-running
        operation to track the change. Newly written documents will have TTLs
        applied as requested. Requested TTLs on existing documents are still
        being processed. When TTLs on all existing documents have been
        processed, the state will move to 'ACTIVE'.
      ACTIVE: The TTL is active for all documents.
      NEEDS_REPAIR: The TTL configuration could not be enabled for all
        existing documents. Newly written documents will continue to have
        their TTL applied. The LRO returned when last attempting to enable TTL
        for this `Field` has failed, and may have more details.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        NEEDS_REPAIR = 3
    state = _messages.EnumField('StateValueValuesEnum', 1)