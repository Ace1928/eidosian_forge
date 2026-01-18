from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpsTrigger(_messages.Message):
    """Describes HttpsTrigger, could be used to connect web hooks to function.

  Enums:
    SecurityLevelValueValuesEnum: The security level for the function.

  Fields:
    securityLevel: The security level for the function.
    url: Output only. The deployed url for the function.
  """

    class SecurityLevelValueValuesEnum(_messages.Enum):
        """The security level for the function.

    Values:
      SECURITY_LEVEL_UNSPECIFIED: Unspecified.
      SECURE_ALWAYS: Requests for a URL that match this handler that do not
        use HTTPS are automatically redirected to the HTTPS URL with the same
        path. Query parameters are reserved for the redirect.
      SECURE_OPTIONAL: Both HTTP and HTTPS requests with URLs that match the
        handler succeed without redirects. The application can examine the
        request to determine which protocol was used and respond accordingly.
    """
        SECURITY_LEVEL_UNSPECIFIED = 0
        SECURE_ALWAYS = 1
        SECURE_OPTIONAL = 2
    securityLevel = _messages.EnumField('SecurityLevelValueValuesEnum', 1)
    url = _messages.StringField(2)