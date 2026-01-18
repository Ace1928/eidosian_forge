from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2HTTPHeader(_messages.Message):
    """HTTPHeader describes a custom header to be used in HTTP probes

  Fields:
    name: Required. The header field name
    value: Optional. The header field value
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)