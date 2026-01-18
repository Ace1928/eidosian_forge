from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class HttpRequestContext(_messages.Message):
    """HTTP request data that is related to a reported error. This data should
  be provided by the application when reporting an error, unless the error
  report has been generated automatically from Google App Engine logs.

  Fields:
    method: The type of HTTP request, such as `GET`, `POST`, etc.
    referrer: The referrer information that is provided with the request.
    remoteIp: The IP address from which the request originated. This can be
      IPv4, IPv6, or a token which is derived from the IP address, depending
      on the data that has been provided in the error report.
    responseStatusCode: The HTTP response status code for the request.
    url: The URL of the request.
    userAgent: The user agent information that is provided with the request.
  """
    method = _messages.StringField(1)
    referrer = _messages.StringField(2)
    remoteIp = _messages.StringField(3)
    responseStatusCode = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    url = _messages.StringField(5)
    userAgent = _messages.StringField(6)