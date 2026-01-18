from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleTypeTimeZone(_messages.Message):
    """Represents a time zone from the [IANA Time Zone
  Database](https://www.iana.org/time-zones).

  Fields:
    id: IANA Time Zone Database time zone, e.g. "America/New_York".
    version: Optional. IANA Time Zone Database version number, e.g. "2019a".
  """
    id = _messages.StringField(1)
    version = _messages.StringField(2)