from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminProxyConfig(_messages.Message):
    """BareMetalAdminProxyConfig specifies the cluster proxy configuration.

  Fields:
    noProxy: A list of IPs, hostnames, and domains that should skip the proxy.
      Examples: ["127.0.0.1", "example.com", ".corp", "localhost"].
    uri: Required. Specifies the address of your proxy server. Examples:
      `http://domain` WARNING: Do not provide credentials in the format
      `http://(username:password@)domain` these will be rejected by the
      server.
  """
    noProxy = _messages.StringField(1, repeated=True)
    uri = _messages.StringField(2)