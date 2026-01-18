from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LanguageLabsAidaTrustRecitationProtoStreamRecitationResult(_messages.Message):
    """The recitation result for one stream input

  Enums:
    RecitationActionValueValuesEnum: The recitation action for one given
      input. When its segments contain different actions, the overall action
      will be returned in the precedence of BLOCK > CITE > NO_ACTION.

  Fields:
    dynamicSegmentResults: The recitation result against the given dynamic
      data source.
    fullyCheckedTextIndex: Last index of input text fully checked for
      recitation in the entire streaming context. Would return `-1` if no
      Input was checked for recitation.
    recitationAction: The recitation action for one given input. When its
      segments contain different actions, the overall action will be returned
      in the precedence of BLOCK > CITE > NO_ACTION.
    trainingSegmentResults: The recitation result against model training data.
  """

    class RecitationActionValueValuesEnum(_messages.Enum):
        """The recitation action for one given input. When its segments contain
    different actions, the overall action will be returned in the precedence
    of BLOCK > CITE > NO_ACTION.

    Values:
      ACTION_UNSPECIFIED: <no description>
      CITE: indicate that attribution must be shown for a Segment
      BLOCK: indicate that a Segment should be blocked from being used
      NO_ACTION: for tagging high-frequency code snippets
      EXEMPT_FOUND_IN_PROMPT: The recitation was found in prompt and is
        exempted from overall results
    """
        ACTION_UNSPECIFIED = 0
        CITE = 1
        BLOCK = 2
        NO_ACTION = 3
        EXEMPT_FOUND_IN_PROMPT = 4
    dynamicSegmentResults = _messages.MessageField('LanguageLabsAidaTrustRecitationProtoSegmentResult', 1, repeated=True)
    fullyCheckedTextIndex = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    recitationAction = _messages.EnumField('RecitationActionValueValuesEnum', 3)
    trainingSegmentResults = _messages.MessageField('LanguageLabsAidaTrustRecitationProtoSegmentResult', 4, repeated=True)