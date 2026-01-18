from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class SniServerTlsPoliciesValue(_messages.Message):
    """Optional. DEPRECATED: DO NOT USE A map from SNI values to server TLS
    policies. The SNI value is a 16-byte (128-bit) ASCII string that
    represents the hostname of the server that the client is trying to connect
    to. The hostname must be a valid hostname, according to the rules of the
    Domain Name System (DNS). This means that the hostname must start with a
    letter or a numeric character, and it can contain any combination of
    letters, numbers, and hyphens. The hostname cannot be longer than 63
    characters, and it cannot end with a hyphen. Note that partial wildcards
    are not supported, and values like `*w.example.com` are invalid. The value
    `*` is also not supported. The ServerTLSPolicy value should be the
    canonical resource name of the ServerTLSPolicy resource being referenced
    for example: `projects//locations/global/serverTlsPolicies/server-tls-
    policy`. If both this field and the server_tls_policy/sni_hosts field are
    set this field will be ignored.

    Messages:
      AdditionalProperty: An additional property for a
        SniServerTlsPoliciesValue object.

    Fields:
      additionalProperties: Additional properties of type
        SniServerTlsPoliciesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a SniServerTlsPoliciesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)