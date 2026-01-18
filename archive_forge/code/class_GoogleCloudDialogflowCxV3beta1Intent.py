from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1Intent(_messages.Message):
    """An intent represents a user's intent to interact with a conversational
  agent. You can provide information for the Dialogflow API to use to match
  user input to an intent by adding training phrases (i.e., examples of user
  input) to your intent.

  Messages:
    LabelsValue: The key/value metadata to label an intent. Labels can contain
      lowercase letters, digits and the symbols '-' and '_'. International
      characters are allowed, including letters from unicase alphabets. Keys
      must start with a letter. Keys and values can be no longer than 63
      characters and no more than 128 bytes. Prefix "sys-" is reserved for
      Dialogflow defined labels. Currently allowed Dialogflow defined labels
      include: * sys-head * sys-contextual The above labels do not require
      value. "sys-head" means the intent is a head intent. "sys-contextual"
      means the intent is a contextual intent.

  Fields:
    description: Human readable description for better understanding an intent
      like its scope, content, result etc. Maximum character limit: 140
      characters.
    displayName: Required. The human-readable name of the intent, unique
      within the agent.
    isFallback: Indicates whether this is a fallback intent. Currently only
      default fallback intent is allowed in the agent, which is added upon
      agent creation. Adding training phrases to fallback intent is useful in
      the case of requests that are mistakenly matched, since training phrases
      assigned to fallback intents act as negative examples that triggers no-
      match event.
    labels: The key/value metadata to label an intent. Labels can contain
      lowercase letters, digits and the symbols '-' and '_'. International
      characters are allowed, including letters from unicase alphabets. Keys
      must start with a letter. Keys and values can be no longer than 63
      characters and no more than 128 bytes. Prefix "sys-" is reserved for
      Dialogflow defined labels. Currently allowed Dialogflow defined labels
      include: * sys-head * sys-contextual The above labels do not require
      value. "sys-head" means the intent is a head intent. "sys-contextual"
      means the intent is a contextual intent.
    name: The unique identifier of the intent. Required for the
      Intents.UpdateIntent method. Intents.CreateIntent populates the name
      automatically. Format: `projects//locations//agents//intents/`.
    parameters: The collection of parameters associated with the intent.
    priority: The priority of this intent. Higher numbers represent higher
      priorities. - If the supplied value is unspecified or 0, the service
      translates the value to 500,000, which corresponds to the `Normal`
      priority in the console. - If the supplied value is negative, the intent
      is ignored in runtime detect intent requests.
    trainingPhrases: The collection of training phrases the agent is trained
      on to identify the intent.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The key/value metadata to label an intent. Labels can contain
    lowercase letters, digits and the symbols '-' and '_'. International
    characters are allowed, including letters from unicase alphabets. Keys
    must start with a letter. Keys and values can be no longer than 63
    characters and no more than 128 bytes. Prefix "sys-" is reserved for
    Dialogflow defined labels. Currently allowed Dialogflow defined labels
    include: * sys-head * sys-contextual The above labels do not require
    value. "sys-head" means the intent is a head intent. "sys-contextual"
    means the intent is a contextual intent.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    isFallback = _messages.BooleanField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    parameters = _messages.MessageField('GoogleCloudDialogflowCxV3beta1IntentParameter', 6, repeated=True)
    priority = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    trainingPhrases = _messages.MessageField('GoogleCloudDialogflowCxV3beta1IntentTrainingPhrase', 8, repeated=True)