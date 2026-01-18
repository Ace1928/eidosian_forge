from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildStreamContentRequest(_messages.Message):
    """Message for building a StreamContent

  Fields:
    contentVersionTag: Required. The user-specified version tag of the build
      if it succeeds. Must match \\w{0,127}. See also https://github.com/distri
      bution/distribution/blob/main/reference/regexp.go
    requestId: Optional. A unique identifier for this request. Restricted to
      36 ASCII characters. A random UUID is recommended. This request is only
      idempotent if a `request_id` is provided."
  """
    contentVersionTag = _messages.StringField(1)
    requestId = _messages.StringField(2)