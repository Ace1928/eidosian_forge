from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2TimeZone(_messages.Message):
    """Time zone of the date time object.

  Fields:
    offsetMinutes: Set only if the offset can be determined. Positive for time
      ahead of UTC. E.g. For "UTC-9", this value is -540.
  """
    offsetMinutes = _messages.IntegerField(1, variant=_messages.Variant.INT32)