from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def GetSslPolicyForPatch(self, fingerprint, profile=None, min_tls_version=None, custom_features=None):
    """Returns the SslPolicy message for a patch request.

    Args:
      fingerprint: String representing the existing fingerprint of the SSL
        policy resource.
      profile: String representing the SSL policy profile. Can be one of
        'COMPATIBLE', 'MODERN', 'RESTRICTED' or 'CUSTOM'.
      min_tls_version: String representing the minimum TLS version of the SSL
        policy. Can be one of 'TLS_1_0', 'TLS_1_1', 'TLS_1_2'.
      custom_features: The list of strings representing the custom features to
        use.
    """
    messages = self._messages
    ssl_policy = messages.SslPolicy(fingerprint=fingerprint)
    if profile:
        ssl_policy.profile = messages.SslPolicy.ProfileValueValuesEnum(profile)
    if min_tls_version:
        ssl_policy.minTlsVersion = messages.SslPolicy.MinTlsVersionValueValuesEnum(min_tls_version)
    if custom_features is not None:
        ssl_policy.customFeatures = custom_features
    return ssl_policy