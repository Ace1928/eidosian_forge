from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta1DocumentRevisionHumanReview(_messages.Message):
    """Human Review information of the document.

  Fields:
    state: Human review state. e.g. `requested`, `succeeded`, `rejected`.
    stateMessage: A message providing more details about the current state of
      processing. For example, the rejection reason when the state is
      `rejected`.
  """
    state = _messages.StringField(1)
    stateMessage = _messages.StringField(2)