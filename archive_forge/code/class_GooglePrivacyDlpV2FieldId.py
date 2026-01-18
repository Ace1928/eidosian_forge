from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2FieldId(_messages.Message):
    """General identifier of a data field in a storage service.

  Fields:
    name: Name describing the field.
  """
    name = _messages.StringField(1)