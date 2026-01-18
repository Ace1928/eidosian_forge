from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3IntentTrainingPhrasePart(_messages.Message):
    """Represents a part of a training phrase.

  Fields:
    parameterId: The parameter used to annotate this part of the training
      phrase. This field is required for annotated parts of the training
      phrase.
    text: Required. The text for this part.
  """
    parameterId = _messages.StringField(1)
    text = _messages.StringField(2)