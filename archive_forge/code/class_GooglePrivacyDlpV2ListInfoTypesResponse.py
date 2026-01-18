from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ListInfoTypesResponse(_messages.Message):
    """Response to the ListInfoTypes request.

  Fields:
    infoTypes: Set of sensitive infoTypes.
  """
    infoTypes = _messages.MessageField('GooglePrivacyDlpV2InfoTypeDescription', 1, repeated=True)