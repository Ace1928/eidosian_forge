from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1TlsInfo(_messages.Message):
    """TLS configuration information for virtual hosts and TargetServers.

  Fields:
    ciphers: The SSL/TLS cipher suites to be used. For programmable proxies,
      it must be one of the cipher suite names listed in: http://docs.oracle.c
      om/javase/8/docs/technotes/guides/security/StandardNames.html#ciphersuit
      es. For configurable proxies, it must follow the configuration specified
      in: https://commondatastorage.googleapis.com/chromium-boringssl-
      docs/ssl.h.html#Cipher-suite-configuration. This setting has no effect
      for configurable proxies when negotiating TLS 1.3.
    clientAuthEnabled: Optional. Enables two-way TLS.
    commonName: The TLS Common Name of the certificate.
    enabled: Required. Enables TLS. If false, neither one-way nor two-way TLS
      will be enabled.
    enforce: TLS is strictly enforced. TODO (b/331425331) remove
      TRUSTED_TESTER when ready for public
    ignoreValidationErrors: If true, Edge ignores TLS certificate errors.
      Valid when configuring TLS for target servers and target endpoints, and
      when configuring virtual hosts that use 2-way TLS. When used with a
      target endpoint/target server, if the backend system uses SNI and
      returns a cert with a subject Distinguished Name (DN) that does not
      match the hostname, there is no way to ignore the error and the
      connection fails.
    keyAlias: Required if `client_auth_enabled` is true. The resource ID for
      the alias containing the private key and cert.
    keyStore: Required if `client_auth_enabled` is true. The resource ID of
      the keystore.
    protocols: The TLS versioins to be used.
    trustStore: The resource ID of the truststore.
  """
    ciphers = _messages.StringField(1, repeated=True)
    clientAuthEnabled = _messages.BooleanField(2)
    commonName = _messages.MessageField('GoogleCloudApigeeV1TlsInfoCommonName', 3)
    enabled = _messages.BooleanField(4)
    enforce = _messages.BooleanField(5)
    ignoreValidationErrors = _messages.BooleanField(6)
    keyAlias = _messages.StringField(7)
    keyStore = _messages.StringField(8)
    protocols = _messages.StringField(9, repeated=True)
    trustStore = _messages.StringField(10)