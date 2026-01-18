from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SuggestionClusterProto(_messages.Message):
    """A set of similar suggestions that we suspect are closely related. This
  proto and most of the nested protos are branched from
  foxandcrown.prelaunchreport.service.SuggestionClusterProto, replacing PLR's
  dependencies with FTL's.

  Enums:
    CategoryValueValuesEnum: Category in which these types of suggestions
      should appear. Always set.

  Fields:
    category: Category in which these types of suggestions should appear.
      Always set.
    suggestions: A sequence of suggestions. All of the suggestions within a
      cluster must have the same SuggestionPriority and belong to the same
      SuggestionCategory. Suggestions with the same screenshot URL should be
      adjacent.
  """

    class CategoryValueValuesEnum(_messages.Enum):
        """Category in which these types of suggestions should appear. Always
    set.

    Values:
      unknownCategory: <no description>
      contentLabeling: <no description>
      touchTargetSize: <no description>
      lowContrast: <no description>
      implementation: <no description>
    """
        unknownCategory = 0
        contentLabeling = 1
        touchTargetSize = 2
        lowContrast = 3
        implementation = 4
    category = _messages.EnumField('CategoryValueValuesEnum', 1)
    suggestions = _messages.MessageField('SuggestionProto', 2, repeated=True)