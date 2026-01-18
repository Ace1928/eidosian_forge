from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelemetryProviderFileAccessLog(_messages.Message):
    """File access log provider configuration. Envoy uses the following default
  format: [%START_TIME%] "%REQ(:METHOD)% %REQ(X-ENVOY-ORIGINAL-PATH?:PATH)%
  %PROTOCOL%" %RESPONSE_CODE% %RESPONSE_FLAGS% %BYTES_RECEIVED% %BYTES_SENT%
  %DURATION% %RESP(X-ENVOY-UPSTREAM-SERVICE-TIME)% "%REQ(X-FORWARDED-FOR)%"
  "%REQ(USER-AGENT)%" "%REQ(X-REQUEST-ID)%" "%REQ(:AUTHORITY)%"
  "%UPSTREAM_HOST%"\\n Example of the default Envoy access log format:
  [2016-04-15T20:17:00.310Z] "POST /api/v1/locations HTTP/2" 204 - 154 0 226
  100 "10.0.35.28" "nsq2http" "cc21d9b0-cf5c-432b-8c7e-98aeb7988cd2"
  "locations" "tcp://10.0.2.1:80"

  Fields:
    filePath: Optional. Path to a local file to write the access log entries.
      This may be used to write to streams, via /dev/stderr and /dev/stdout.
      If unspecified, defaults to /dev/stdout.
  """
    filePath = _messages.StringField(1)