from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1IntentTrainingPhrase(_messages.Message):
    """Represents an example that the agent is trained on to identify the
  intent.

  Fields:
    id: Output only. The unique identifier of the training phrase.
    parts: Required. The ordered list of training phrase parts. The parts are
      concatenated in order to form the training phrase. Note: The API does
      not automatically annotate training phrases like the Dialogflow Console
      does. Note: Do not forget to include whitespace at part boundaries, so
      the training phrase is well formatted when the parts are concatenated.
      If the training phrase does not need to be annotated with parameters,
      you just need a single part with only the Part.text field set. If you
      want to annotate the training phrase, you must create multiple parts,
      where the fields of each part are populated in one of two ways: -
      `Part.text` is set to a part of the phrase that has no parameters. -
      `Part.text` is set to a part of the phrase that you want to annotate,
      and the `parameter_id` field is set.
    repeatCount: Indicates how many times this example was added to the
      intent.
  """
    id = _messages.StringField(1)
    parts = _messages.MessageField('GoogleCloudDialogflowCxV3beta1IntentTrainingPhrasePart', 2, repeated=True)
    repeatCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)