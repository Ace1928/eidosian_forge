from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Row(_messages.Message):
    """Values of the row.

  Fields:
    values: Individual cells.
  """
    values = _messages.MessageField('GooglePrivacyDlpV2Value', 1, repeated=True)