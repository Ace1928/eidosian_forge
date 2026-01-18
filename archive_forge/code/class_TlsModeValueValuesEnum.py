from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TlsModeValueValuesEnum(_messages.Enum):
    """Indicates whether connections should be secured using TLS. The value
    of this field determines how TLS is enforced. This field can be set to one
    of the following: - SIMPLE Secure connections with standard TLS semantics.
    - MUTUAL Secure connections to the backends using mutual TLS by presenting
    client certificates for authentication.

    Values:
      INVALID: <no description>
      MUTUAL: Secure connections to the backends using mutual TLS by
        presenting client certificates for authentication.
      SIMPLE: Secure connections with standard TLS semantics.
    """
    INVALID = 0
    MUTUAL = 1
    SIMPLE = 2