from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StaticFilesHandler(_messages.Message):
    """Files served directly to the user for a given URL, such as images, CSS
  stylesheets, or JavaScript source files. Static file handlers describe which
  files in the application directory are static files, and which URLs serve
  them.

  Messages:
    HttpHeadersValue: HTTP headers to use for all responses from these URLs.

  Fields:
    applicationReadable: Whether files should also be uploaded as code data.
      By default, files declared in static file handlers are uploaded as
      static data and are only served to end users; they cannot be read by the
      application. If enabled, uploads are charged against both your code and
      static data storage resource quotas.
    expiration: Time a static file served by this handler should be cached by
      web proxies and browsers.
    httpHeaders: HTTP headers to use for all responses from these URLs.
    mimeType: MIME type used to serve all files served by this
      handler.Defaults to file-specific MIME types, which are derived from
      each file's filename extension.
    path: Path to the static files matched by the URL pattern, from the
      application root directory. The path can refer to text matched in
      groupings in the URL pattern.
    requireMatchingFile: Whether this handler should match the request if the
      file referenced by the handler does not exist.
    uploadPathRegex: Regular expression that matches the file paths for all
      files that should be referenced by this handler.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class HttpHeadersValue(_messages.Message):
        """HTTP headers to use for all responses from these URLs.

    Messages:
      AdditionalProperty: An additional property for a HttpHeadersValue
        object.

    Fields:
      additionalProperties: Additional properties of type HttpHeadersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a HttpHeadersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    applicationReadable = _messages.BooleanField(1)
    expiration = _messages.StringField(2)
    httpHeaders = _messages.MessageField('HttpHeadersValue', 3)
    mimeType = _messages.StringField(4)
    path = _messages.StringField(5)
    requireMatchingFile = _messages.BooleanField(6)
    uploadPathRegex = _messages.StringField(7)