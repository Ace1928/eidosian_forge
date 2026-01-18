from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomDns(_messages.Message):
    """Configuration for an arbitrary DNS provider.

  Fields:
    dsRecords: The list of DS records for this domain, which are used to
      enable DNSSEC. The domain's DNS provider can provide the values to set
      here. If this field is empty, DNSSEC is disabled.
    nameServers: Required. A list of name servers that store the DNS zone for
      this domain. Each name server is a domain name, with Unicode domain
      names expressed in Punycode format.
  """
    dsRecords = _messages.MessageField('DsRecord', 1, repeated=True)
    nameServers = _messages.StringField(2, repeated=True)