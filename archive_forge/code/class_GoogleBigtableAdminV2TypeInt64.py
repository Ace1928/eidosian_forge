from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleBigtableAdminV2TypeInt64(_messages.Message):
    """Int64 Values of type `Int64` are stored in `Value.int_value`.

  Fields:
    encoding: The encoding to use when converting to/from lower level types.
  """
    encoding = _messages.MessageField('GoogleBigtableAdminV2TypeInt64Encoding', 1)