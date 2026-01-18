from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3AdvancedSettingsDtmfSettings(_messages.Message):
    """Define behaviors for DTMF (dual tone multi frequency).

  Fields:
    enabled: If true, incoming audio is processed for DTMF (dual tone multi
      frequency) events. For example, if the caller presses a button on their
      telephone keypad and DTMF processing is enabled, Dialogflow will detect
      the event (e.g. a "3" was pressed) in the incoming audio and pass the
      event to the bot to drive business logic (e.g. when 3 is pressed, return
      the account balance).
    finishDigit: The digit that terminates a DTMF digit sequence.
    maxDigits: Max length of DTMF digits.
  """
    enabled = _messages.BooleanField(1)
    finishDigit = _messages.StringField(2)
    maxDigits = _messages.IntegerField(3, variant=_messages.Variant.INT32)