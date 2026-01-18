from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SuggestionProto(_messages.Message):
    """A SuggestionProto object.

  Enums:
    PriorityValueValuesEnum: Relative importance of a suggestion. Always set.

  Fields:
    helpUrl: Reference to a help center article concerning this type of
      suggestion. Always set.
    longMessage: Message, in the user's language, explaining the suggestion,
      which may contain markup. Always set.
    priority: Relative importance of a suggestion. Always set.
    pseudoResourceId: A somewhat human readable identifier of the source view,
      if it does not have a resource_name. This is a path within the
      accessibility hierarchy, an element with resource name; similar to an
      XPath.
    region: Region within the screenshot that is relevant to this suggestion.
      Optional.
    resourceName: Reference to a view element, identified by its resource
      name, if it has one.
    screenId: ID of the screen for the suggestion. It is used for getting the
      corresponding screenshot path. For example, screen_id "1" corresponds to
      "1.png" file in GCS. Always set.
    secondaryPriority: Relative importance of a suggestion as compared with
      other suggestions that have the same priority and category. This is a
      meaningless value that can be used to order suggestions that are in the
      same category and have the same priority. The larger values have higher
      priority (i.e., are more important). Optional.
    shortMessage: Concise message, in the user's language, representing the
      suggestion, which may contain markup. Always set.
    title: General title for the suggestion, in the user's language, without
      markup. Always set.
  """

    class PriorityValueValuesEnum(_messages.Enum):
        """Relative importance of a suggestion. Always set.

    Values:
      unknownPriority: <no description>
      error: <no description>
      warning: <no description>
      info: <no description>
    """
        unknownPriority = 0
        error = 1
        warning = 2
        info = 3
    helpUrl = _messages.StringField(1)
    longMessage = _messages.MessageField('SafeHtmlProto', 2)
    priority = _messages.EnumField('PriorityValueValuesEnum', 3)
    pseudoResourceId = _messages.StringField(4)
    region = _messages.MessageField('RegionProto', 5)
    resourceName = _messages.StringField(6)
    screenId = _messages.StringField(7)
    secondaryPriority = _messages.FloatField(8)
    shortMessage = _messages.MessageField('SafeHtmlProto', 9)
    title = _messages.StringField(10)