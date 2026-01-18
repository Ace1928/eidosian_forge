from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TlsFeatureProfileValueValuesEnum(_messages.Enum):
    """Optional. The selected Profile. If this is not set, then the default
    value is to allow the broadest set of clients and servers
    ("PROFILE_COMPATIBLE"). Setting this to more restrictive values may
    improve security, but may also prevent the TLS inspection proxy from
    connecting to some clients or servers. Note that Secure Web Proxy does not
    yet honor this field.

    Values:
      PROFILE_UNSPECIFIED: Indicates no profile was specified.
      PROFILE_COMPATIBLE: Compatible profile. Allows the broadest set of
        clients, even those which support only out-of-date SSL features to
        negotiate with the TLS inspection proxy.
      PROFILE_MODERN: Modern profile. Supports a wide set of SSL features,
        allowing modern clients to negotiate SSL with the TLS inspection
        proxy.
      PROFILE_RESTRICTED: Restricted profile. Supports a reduced set of SSL
        features, intended to meet stricter compliance requirements.
      PROFILE_CUSTOM: Custom profile. Allow only the set of allowed SSL
        features specified in the custom_features field of SslPolicy.
    """
    PROFILE_UNSPECIFIED = 0
    PROFILE_COMPATIBLE = 1
    PROFILE_MODERN = 2
    PROFILE_RESTRICTED = 3
    PROFILE_CUSTOM = 4