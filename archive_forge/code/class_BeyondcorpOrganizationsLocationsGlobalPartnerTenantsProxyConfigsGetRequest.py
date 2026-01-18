from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpOrganizationsLocationsGlobalPartnerTenantsProxyConfigsGetRequest(_messages.Message):
    """A
  BeyondcorpOrganizationsLocationsGlobalPartnerTenantsProxyConfigsGetRequest
  object.

  Fields:
    name: Required. The resource name of the ProxyConfig using the form: `orga
      nizations/{organization_id}/locations/global/partnerTenants/{partner_ten
      ant_id}/proxyConfigs/{proxy_config_id}`
  """
    name = _messages.StringField(1, required=True)