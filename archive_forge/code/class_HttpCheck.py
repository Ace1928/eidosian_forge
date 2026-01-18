from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpCheck(_messages.Message):
    """Information involved in an HTTP/HTTPS Uptime check request.

  Enums:
    ContentTypeValueValuesEnum: The content type header to use for the check.
      The following configurations result in errors: 1. Content type is
      specified in both the headers field and the content_type field. 2.
      Request method is GET and content_type is not TYPE_UNSPECIFIED 3.
      Request method is POST and content_type is TYPE_UNSPECIFIED. 4. Request
      method is POST and a "Content-Type" header is provided via headers
      field. The content_type field should be used instead.
    RequestMethodValueValuesEnum: The HTTP request method to use for the
      check. If set to METHOD_UNSPECIFIED then request_method defaults to GET.

  Messages:
    HeadersValue: The list of headers to send as part of the Uptime check
      request. If two headers have the same key and different values, they
      should be entered as a single header, with the value being a comma-
      separated list of all the desired values as described at
      https://www.w3.org/Protocols/rfc2616/rfc2616.txt (page 31). Entering two
      separate headers with the same key in a Create call will cause the first
      to be overwritten by the second. The maximum number of headers allowed
      is 100.

  Fields:
    acceptedResponseStatusCodes: If present, the check will only pass if the
      HTTP response status code is in this set of status codes. If empty, the
      HTTP status code will only pass if the HTTP status code is 200-299.
    authInfo: The authentication information. Optional when creating an HTTP
      check; defaults to empty.
    body: The request body associated with the HTTP POST request. If
      content_type is URL_ENCODED, the body passed in must be URL-encoded.
      Users can provide a Content-Length header via the headers field or the
      API will do so. If the request_method is GET and body is not empty, the
      API will return an error. The maximum byte size is 1 megabyte.Note: If
      client libraries aren't used (which performs the conversion
      automatically) base64 encode your body data since the field is of bytes
      type.
    contentType: The content type header to use for the check. The following
      configurations result in errors: 1. Content type is specified in both
      the headers field and the content_type field. 2. Request method is GET
      and content_type is not TYPE_UNSPECIFIED 3. Request method is POST and
      content_type is TYPE_UNSPECIFIED. 4. Request method is POST and a
      "Content-Type" header is provided via headers field. The content_type
      field should be used instead.
    customContentType: A user provided content type header to use for the
      check. The invalid configurations outlined in the content_type field
      apply to custom_content_type, as well as the following: 1. content_type
      is URL_ENCODED and custom_content_type is set. 2. content_type is
      USER_PROVIDED and custom_content_type is not set.
    headers: The list of headers to send as part of the Uptime check request.
      If two headers have the same key and different values, they should be
      entered as a single header, with the value being a comma-separated list
      of all the desired values as described at
      https://www.w3.org/Protocols/rfc2616/rfc2616.txt (page 31). Entering two
      separate headers with the same key in a Create call will cause the first
      to be overwritten by the second. The maximum number of headers allowed
      is 100.
    maskHeaders: Boolean specifying whether to encrypt the header information.
      Encryption should be specified for any headers related to authentication
      that you do not wish to be seen when retrieving the configuration. The
      server will be responsible for encrypting the headers. On Get/List
      calls, if mask_headers is set to true then the headers will be obscured
      with ******.
    path: Optional (defaults to "/"). The path to the page against which to
      run the check. Will be combined with the host (specified within the
      monitored_resource) and port to construct the full URL. If the provided
      path does not begin with "/", a "/" will be prepended automatically.
    pingConfig: Contains information needed to add pings to an HTTP check.
    port: Optional (defaults to 80 when use_ssl is false, and 443 when use_ssl
      is true). The TCP port on the HTTP server against which to run the
      check. Will be combined with host (specified within the
      monitored_resource) and path to construct the full URL.
    requestMethod: The HTTP request method to use for the check. If set to
      METHOD_UNSPECIFIED then request_method defaults to GET.
    useSsl: If true, use HTTPS instead of HTTP to run the check.
    validateSsl: Boolean specifying whether to include SSL certificate
      validation as a part of the Uptime check. Only applies to checks where
      monitored_resource is set to uptime_url. If use_ssl is false, setting
      validate_ssl to true has no effect.
  """

    class ContentTypeValueValuesEnum(_messages.Enum):
        """The content type header to use for the check. The following
    configurations result in errors: 1. Content type is specified in both the
    headers field and the content_type field. 2. Request method is GET and
    content_type is not TYPE_UNSPECIFIED 3. Request method is POST and
    content_type is TYPE_UNSPECIFIED. 4. Request method is POST and a
    "Content-Type" header is provided via headers field. The content_type
    field should be used instead.

    Values:
      TYPE_UNSPECIFIED: No content type specified.
      URL_ENCODED: body is in URL-encoded form. Equivalent to setting the
        Content-Type to application/x-www-form-urlencoded in the HTTP request.
      USER_PROVIDED: body is in custom_content_type form. Equivalent to
        setting the Content-Type to the contents of custom_content_type in the
        HTTP request.
    """
        TYPE_UNSPECIFIED = 0
        URL_ENCODED = 1
        USER_PROVIDED = 2

    class RequestMethodValueValuesEnum(_messages.Enum):
        """The HTTP request method to use for the check. If set to
    METHOD_UNSPECIFIED then request_method defaults to GET.

    Values:
      METHOD_UNSPECIFIED: No request method specified.
      GET: GET request.
      POST: POST request.
    """
        METHOD_UNSPECIFIED = 0
        GET = 1
        POST = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class HeadersValue(_messages.Message):
        """The list of headers to send as part of the Uptime check request. If
    two headers have the same key and different values, they should be entered
    as a single header, with the value being a comma-separated list of all the
    desired values as described at
    https://www.w3.org/Protocols/rfc2616/rfc2616.txt (page 31). Entering two
    separate headers with the same key in a Create call will cause the first
    to be overwritten by the second. The maximum number of headers allowed is
    100.

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
    acceptedResponseStatusCodes = _messages.MessageField('ResponseStatusCode', 1, repeated=True)
    authInfo = _messages.MessageField('BasicAuthentication', 2)
    body = _messages.BytesField(3)
    contentType = _messages.EnumField('ContentTypeValueValuesEnum', 4)
    customContentType = _messages.StringField(5)
    headers = _messages.MessageField('HeadersValue', 6)
    maskHeaders = _messages.BooleanField(7)
    path = _messages.StringField(8)
    pingConfig = _messages.MessageField('PingConfig', 9)
    port = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    requestMethod = _messages.EnumField('RequestMethodValueValuesEnum', 11)
    useSsl = _messages.BooleanField(12)
    validateSsl = _messages.BooleanField(13)