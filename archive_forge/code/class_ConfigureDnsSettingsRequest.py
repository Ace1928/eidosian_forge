from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigureDnsSettingsRequest(_messages.Message):
    """Request for the `ConfigureDnsSettings` method.

  Fields:
    dnsSettings: Fields of the `DnsSettings` to update.
    updateMask: Required. The field mask describing which fields to update as
      a comma-separated list. For example, if only the name servers are being
      updated for an existing Custom DNS configuration, the `update_mask` is
      `"custom_dns.name_servers"`. When changing the DNS provider from one
      type to another, pass the new provider's field name as part of the field
      mask. For example, when changing from a Google Domains DNS configuration
      to a Custom DNS configuration, the `update_mask` is `"custom_dns"`. //
    validateOnly: Validate the request without actually updating the DNS
      settings.
  """
    dnsSettings = _messages.MessageField('DnsSettings', 1)
    updateMask = _messages.StringField(2)
    validateOnly = _messages.BooleanField(3)