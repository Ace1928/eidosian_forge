from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MinTlsVersionValueValuesEnum(_messages.Enum):
    """Optional. Minimum TLS version that the firewall should use when
    negotiating connections with both clients and servers. If this is not set,
    then the default value is to allow the broadest set of clients and servers
    (TLS 1.0 or higher). Setting this to more restrictive values may improve
    security, but may also prevent the firewall from connecting to some
    clients or servers. Note that Secure Web Proxy does not yet honor this
    field.

    Values:
      TLS_VERSION_UNSPECIFIED: Indicates no TLS version was specified.
      TLS_1_0: TLS 1.0
      TLS_1_1: TLS 1.1
      TLS_1_2: TLS 1.2
      TLS_1_3: TLS 1.3
    """
    TLS_VERSION_UNSPECIFIED = 0
    TLS_1_0 = 1
    TLS_1_1 = 2
    TLS_1_2 = 3
    TLS_1_3 = 4