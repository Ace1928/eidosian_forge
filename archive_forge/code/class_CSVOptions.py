from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CSVOptions(_messages.Message):
    """Options to configure CSV formatted reports.

  Fields:
    delimiter: Delimiter characters in CSV.
    headerRequired: If set, will include a header row in the CSV report.
    recordSeparator: Record separator characters in CSV.
  """
    delimiter = _messages.StringField(1)
    headerRequired = _messages.BooleanField(2)
    recordSeparator = _messages.StringField(3)