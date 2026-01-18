from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiHttpBody(_messages.Message):
    """Message that represents an arbitrary HTTP body. It should only be used
  for payload formats that can't be represented as JSON, such as raw binary or
  an HTML page. This message can be used both in streaming and non-streaming
  API methods in the request as well as the response. It can be used as a top-
  level request field, which is convenient if one wants to extract parameters
  from either the URL or HTTP template into the request fields and also want
  access to the raw HTTP body. Example: message GetResourceRequest { // A
  unique request id. string request_id = 1; // The raw HTTP body is bound to
  this field. google.api.HttpBody http_body = 2; } service ResourceService {
  rpc GetResource(GetResourceRequest) returns (google.api.HttpBody); rpc
  UpdateResource(google.api.HttpBody) returns (google.protobuf.Empty); }
  Example with streaming methods: service CaldavService { rpc
  GetCalendar(stream google.api.HttpBody) returns (stream
  google.api.HttpBody); rpc UpdateCalendar(stream google.api.HttpBody) returns
  (stream google.api.HttpBody); } Use of this type only changes how the
  request and response bodies are handled, all other features will continue to
  work unchanged.

  Messages:
    ExtensionsValueListEntry: A ExtensionsValueListEntry object.

  Fields:
    contentType: The HTTP Content-Type header value specifying the content
      type of the body.
    data: The HTTP request/response body as raw binary.
    extensions: Application specific response metadata. Must be set in the
      first response for streaming APIs.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ExtensionsValueListEntry(_messages.Message):
        """A ExtensionsValueListEntry object.

    Messages:
      AdditionalProperty: An additional property for a
        ExtensionsValueListEntry object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ExtensionsValueListEntry object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    contentType = _messages.StringField(1)
    data = _messages.BytesField(2)
    extensions = _messages.MessageField('ExtensionsValueListEntry', 3, repeated=True)