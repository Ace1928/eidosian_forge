from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoModule(_messages.Message):
    """GoModule represents a Go module.

  Fields:
    createTime: Output only. The time when the Go module is created.
    name: The resource name of a Go module.
    updateTime: Output only. The time when the Go module is updated.
    version: The version of the Go module. Must be a valid canonical version
      as defined in https://go.dev/ref/mod#glos-canonical-version.
  """
    createTime = _messages.StringField(1)
    name = _messages.StringField(2)
    updateTime = _messages.StringField(3)
    version = _messages.StringField(4)