from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DateShiftField(_messages.Message):
    """Shift the date by a randomized number of days. See [date
  shifting](https://cloud.google.com/dlp/docs/concepts-date-shifting) for more
  information. Supported [types](https://www.hl7.org/fhir/datatypes.html):
  Date, DateTime.
  """