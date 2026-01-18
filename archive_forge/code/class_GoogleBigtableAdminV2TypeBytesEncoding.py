from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleBigtableAdminV2TypeBytesEncoding(_messages.Message):
    """Rules used to convert to/from lower level types.

  Fields:
    raw: Use `Raw` encoding.
  """
    raw = _messages.MessageField('GoogleBigtableAdminV2TypeBytesEncodingRaw', 1)