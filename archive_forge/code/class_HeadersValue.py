from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class HeadersValue(_messages.Message):
    """HTTP request headers. This map contains the header field names and
    values. Headers can be set when the task is created. These headers
    represent a subset of the headers that will accompany the task's HTTP
    request. Some HTTP request headers will be ignored or replaced. A partial
    list of headers that will be ignored or replaced is: * Host: This will be
    computed by Cloud Tasks and derived from HttpRequest.url. * Content-
    Length: This will be computed by Cloud Tasks. * User-Agent: This will be
    set to `"Google-Cloud-Tasks"`. * `X-Google-*`: Google use only. *
    `X-AppEngine-*`: Google use only. `Content-Type` won't be set by Cloud
    Tasks. You can explicitly set `Content-Type` to a media type when the task
    is created. For example, `Content-Type` can be set to `"application/octet-
    stream"` or `"application/json"`. Headers which can have multiple values
    (according to RFC2616) can be specified using comma-separated values. The
    size of the headers must be less than 80KB.

    Messages:
      AdditionalProperty: An additional property for a HeadersValue object.

    Fields:
      additionalProperties: Additional properties of type HeadersValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a HeadersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)