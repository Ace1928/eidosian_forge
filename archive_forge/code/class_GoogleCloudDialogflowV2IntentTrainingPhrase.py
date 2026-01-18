from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentTrainingPhrase(_messages.Message):
    """Represents an example that the agent is trained on.

  Enums:
    TypeValueValuesEnum: Required. The type of the training phrase.

  Fields:
    name: Output only. The unique identifier of this training phrase.
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
      and the `entity_type`, `alias`, and `user_defined` fields are all set.
    timesAddedCount: Optional. Indicates how many times this example was added
      to the intent. Each time a developer adds an existing sample by editing
      an intent or training, this counter is increased.
    type: Required. The type of the training phrase.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Required. The type of the training phrase.

    Values:
      TYPE_UNSPECIFIED: Not specified. This value should never be used.
      EXAMPLE: Examples do not contain @-prefixed entity type names, but
        example parts can be annotated with entity types.
      TEMPLATE: Templates are not annotated with entity types, but they can
        contain @-prefixed entity type names as substrings. Template mode has
        been deprecated. Example mode is the only supported way to create new
        training phrases. If you have existing training phrases that you've
        created in template mode, those will continue to work.
    """
        TYPE_UNSPECIFIED = 0
        EXAMPLE = 1
        TEMPLATE = 2
    name = _messages.StringField(1)
    parts = _messages.MessageField('GoogleCloudDialogflowV2IntentTrainingPhrasePart', 2, repeated=True)
    timesAddedCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    type = _messages.EnumField('TypeValueValuesEnum', 4)