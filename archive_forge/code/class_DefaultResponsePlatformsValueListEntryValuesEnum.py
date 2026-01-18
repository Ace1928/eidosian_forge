from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DefaultResponsePlatformsValueListEntryValuesEnum(_messages.Enum):
    """DefaultResponsePlatformsValueListEntryValuesEnum enum type.

    Values:
      PLATFORM_UNSPECIFIED: Not specified.
      FACEBOOK: Facebook.
      SLACK: Slack.
      TELEGRAM: Telegram.
      KIK: Kik.
      SKYPE: Skype.
      LINE: Line.
      VIBER: Viber.
      ACTIONS_ON_GOOGLE: Google Assistant See [Dialogflow webhook format](http
        s://developers.google.com/assistant/actions/build/json/dialogflow-
        webhook-json)
      TELEPHONY: Telephony Gateway.
      GOOGLE_HANGOUTS: Google Hangouts.
    """
    PLATFORM_UNSPECIFIED = 0
    FACEBOOK = 1
    SLACK = 2
    TELEGRAM = 3
    KIK = 4
    SKYPE = 5
    LINE = 6
    VIBER = 7
    ACTIONS_ON_GOOGLE = 8
    TELEPHONY = 9
    GOOGLE_HANGOUTS = 10