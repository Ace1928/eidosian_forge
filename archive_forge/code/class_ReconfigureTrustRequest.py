from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReconfigureTrustRequest(_messages.Message):
    """Request message for ReconfigureTrust

  Fields:
    targetDnsIpAddresses: Required. The target DNS server IP addresses to
      resolve the remote domain involved in the trust.
    targetDomainName: Required. The fully-qualified target domain name which
      will be in trust with current domain.
  """
    targetDnsIpAddresses = _messages.StringField(1, repeated=True)
    targetDomainName = _messages.StringField(2)