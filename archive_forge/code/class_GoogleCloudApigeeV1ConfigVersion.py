from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ConfigVersion(_messages.Message):
    """Version of the API proxy configuration schema. Currently, only 4.0 is
  supported.

  Fields:
    majorVersion: Major version of the API proxy configuration schema.
    minorVersion: Minor version of the API proxy configuration schema.
  """
    majorVersion = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    minorVersion = _messages.IntegerField(2, variant=_messages.Variant.INT32)