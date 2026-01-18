from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DestinationDetails(_messages.Message):
    """Represents the locations where the generated reports is saved.

  Fields:
    gcsBucketUri: The Cloud Storage bucket where the audit report is/will be
      uploaded.
  """
    gcsBucketUri = _messages.StringField(1)