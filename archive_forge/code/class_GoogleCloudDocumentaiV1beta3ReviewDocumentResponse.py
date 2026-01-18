from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta3ReviewDocumentResponse(_messages.Message):
    """Response message for the ReviewDocument method.

  Enums:
    StateValueValuesEnum: The state of the review operation.

  Fields:
    gcsDestination: The Cloud Storage uri for the human reviewed document if
      the review is succeeded.
    rejectionReason: The reason why the review is rejected by reviewer.
    state: The state of the review operation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of the review operation.

    Values:
      STATE_UNSPECIFIED: The default value. This value is used if the state is
        omitted.
      REJECTED: The review operation is rejected by the reviewer.
      SUCCEEDED: The review operation is succeeded.
    """
        STATE_UNSPECIFIED = 0
        REJECTED = 1
        SUCCEEDED = 2
    gcsDestination = _messages.StringField(1)
    rejectionReason = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)