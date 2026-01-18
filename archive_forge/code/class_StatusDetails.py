from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StatusDetails(_messages.Message):
    """StatusDetails is a set of additional properties that MAY be set by the
  server to provide additional information about a response. The Reason field
  of a Status object defines what attributes will be set. Clients must ignore
  fields that do not match the defined type of each attribute, and should
  assume that any attribute may be empty, invalid, or under defined.

  Fields:
    causes: The Causes array includes more details associated with the
      StatusReason failure. Not all StatusReasons may provide detailed causes.
    group: The group attribute of the resource associated with the status
      StatusReason.
    kind: The kind attribute of the resource associated with the status
      StatusReason. On some operations may differ from the requested resource
      Kind.
    name: The name attribute of the resource associated with the status
      StatusReason (when there is a single name which can be described).
    retryAfterSeconds: If specified, the time in seconds before the operation
      should be retried. Some errors may indicate the client must take an
      alternate action - for those errors this field may indicate how long to
      wait before taking the alternate action.
    uid: UID of the resource. (when there is a single resource which can be
      described).
  """
    causes = _messages.MessageField('StatusCause', 1, repeated=True)
    group = _messages.StringField(2)
    kind = _messages.StringField(3)
    name = _messages.StringField(4)
    retryAfterSeconds = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    uid = _messages.StringField(6)