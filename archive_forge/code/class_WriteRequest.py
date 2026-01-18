from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WriteRequest(_messages.Message):
    """The request for Firestore.Write. The first request creates a stream, or
  resumes an existing one from a token. When creating a new stream, the server
  replies with a response containing only an ID and a token, to use in the
  next request. When resuming a stream, the server first streams any responses
  later than the given token, then a response containing only an up-to-date
  token, to use in the next request.

  Messages:
    LabelsValue: Labels associated with this write request.

  Fields:
    labels: Labels associated with this write request.
    streamId: The ID of the write stream to resume. This may only be set in
      the first message. When left empty, a new write stream will be created.
    streamToken: A stream token that was previously sent by the server. The
      client should set this field to the token from the most recent
      WriteResponse it has received. This acknowledges that the client has
      received responses up to this token. After sending this token, earlier
      tokens may not be used anymore. The server may close the stream if there
      are too many unacknowledged responses. Leave this field unset when
      creating a new stream. To resume a stream at a specific point, set this
      field and the `stream_id` field. Leave this field unset when creating a
      new stream.
    writes: The writes to apply. Always executed atomically and in order. This
      must be empty on the first request. This may be empty on the last
      request. This must not be empty on all other requests.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels associated with this write request.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    labels = _messages.MessageField('LabelsValue', 1)
    streamId = _messages.StringField(2)
    streamToken = _messages.BytesField(3)
    writes = _messages.MessageField('Write', 4, repeated=True)