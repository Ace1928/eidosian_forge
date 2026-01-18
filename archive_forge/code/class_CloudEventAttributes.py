from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudEventAttributes(_messages.Message):
    """From
  https://github.com/knative/pkg/blob/master/apis/duck/v1/source_types.go
  CloudEventAttributes are the specific attributes that the Source uses as
  part of its CloudEvents.

  Fields:
    source: Source is the CloudEvents source attribute.
    type: Type refers to the CloudEvent type attribute.
  """
    source = _messages.StringField(1)
    type = _messages.StringField(2)