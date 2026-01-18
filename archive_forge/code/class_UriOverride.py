from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UriOverride(_messages.Message):
    """URI Override. When specified, all the HTTP tasks inside the queue will
  be partially or fully overridden depending on the configured values.

  Enums:
    SchemeValueValuesEnum: Scheme override. When specified, the task URI
      scheme is replaced by the provided value (HTTP or HTTPS).
    UriOverrideEnforceModeValueValuesEnum: URI Override Enforce Mode When
      specified, determines the Target UriOverride mode. If not specified, it
      defaults to ALWAYS.

  Fields:
    host: Host override. When specified, replaces the host part of the task
      URL. For example, if the task URL is "https://www.google.com," and host
      value is set to "example.net", the overridden URI will be changed to
      "https://example.net." Host value cannot be an empty string
      (INVALID_ARGUMENT).
    pathOverride: URI path. When specified, replaces the existing path of the
      task URL. Setting the path value to an empty string clears the URI path
      segment.
    port: Port override. When specified, replaces the port part of the task
      URI. For instance, for a URI http://www.google.com/foo and port=123, the
      overridden URI becomes http://www.google.com:123/foo. Note that the port
      value must be a positive integer. Setting the port to 0 (Zero) clears
      the URI port.
    queryOverride: URI query. When specified, replaces the query part of the
      task URI. Setting the query value to an empty string clears the URI
      query segment.
    scheme: Scheme override. When specified, the task URI scheme is replaced
      by the provided value (HTTP or HTTPS).
    uriOverrideEnforceMode: URI Override Enforce Mode When specified,
      determines the Target UriOverride mode. If not specified, it defaults to
      ALWAYS.
  """

    class SchemeValueValuesEnum(_messages.Enum):
        """Scheme override. When specified, the task URI scheme is replaced by
    the provided value (HTTP or HTTPS).

    Values:
      SCHEME_UNSPECIFIED: Scheme unspecified. Defaults to HTTPS.
      HTTP: Convert the scheme to HTTP, e.g., https://www.google.ca will
        change to http://www.google.ca.
      HTTPS: Convert the scheme to HTTPS, e.g., http://www.google.ca will
        change to https://www.google.ca.
    """
        SCHEME_UNSPECIFIED = 0
        HTTP = 1
        HTTPS = 2

    class UriOverrideEnforceModeValueValuesEnum(_messages.Enum):
        """URI Override Enforce Mode When specified, determines the Target
    UriOverride mode. If not specified, it defaults to ALWAYS.

    Values:
      URI_OVERRIDE_ENFORCE_MODE_UNSPECIFIED: UriOverrideEnforceMode
        Unspecified. Defaults to ALWAYS.
      IF_NOT_EXISTS: In the IF_NOT_EXISTS mode, queue-level configuration is
        only applied where task-level configuration does not exist.
      ALWAYS: In the ALWAYS mode, queue-level configuration overrides all
        task-level configuration
    """
        URI_OVERRIDE_ENFORCE_MODE_UNSPECIFIED = 0
        IF_NOT_EXISTS = 1
        ALWAYS = 2
    host = _messages.StringField(1)
    pathOverride = _messages.MessageField('PathOverride', 2)
    port = _messages.IntegerField(3)
    queryOverride = _messages.MessageField('QueryOverride', 4)
    scheme = _messages.EnumField('SchemeValueValuesEnum', 5)
    uriOverrideEnforceMode = _messages.EnumField('UriOverrideEnforceModeValueValuesEnum', 6)