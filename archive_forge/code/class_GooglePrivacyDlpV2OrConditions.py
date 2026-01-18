from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2OrConditions(_messages.Message):
    """There is an OR relationship between these attributes. They are used to
  determine if a table should be scanned or not in Discovery.

  Fields:
    minAge: Minimum age a table must have before Cloud DLP can profile it.
      Value must be 1 hour or greater.
    minRowCount: Minimum number of rows that should be present before Cloud
      DLP profiles a table
  """
    minAge = _messages.StringField(1)
    minRowCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)