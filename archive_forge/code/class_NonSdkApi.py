from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NonSdkApi(_messages.Message):
    """A non-sdk API and examples of it being called along with other metadata
  See https://developer.android.com/distribute/best-
  practices/develop/restrictions-non-sdk-interfaces

  Enums:
    ListValueValuesEnum: Which list this API appears on

  Fields:
    apiSignature: The signature of the Non-SDK API
    exampleStackTraces: Example stack traces of this API being called.
    insights: Optional debugging insights for non-SDK API violations.
    invocationCount: The total number of times this API was observed to have
      been called.
    list: Which list this API appears on
  """

    class ListValueValuesEnum(_messages.Enum):
        """Which list this API appears on

    Values:
      NONE: <no description>
      WHITE: <no description>
      BLACK: <no description>
      GREY: <no description>
      GREY_MAX_O: <no description>
      GREY_MAX_P: <no description>
      GREY_MAX_Q: <no description>
      GREY_MAX_R: <no description>
      GREY_MAX_S: <no description>
    """
        NONE = 0
        WHITE = 1
        BLACK = 2
        GREY = 3
        GREY_MAX_O = 4
        GREY_MAX_P = 5
        GREY_MAX_Q = 6
        GREY_MAX_R = 7
        GREY_MAX_S = 8
    apiSignature = _messages.StringField(1)
    exampleStackTraces = _messages.StringField(2, repeated=True)
    insights = _messages.MessageField('NonSdkApiInsight', 3, repeated=True)
    invocationCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    list = _messages.EnumField('ListValueValuesEnum', 5)