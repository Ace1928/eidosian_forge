from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesGetRequest(_messages.Message):
    """A BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesGet
  Request object.

  Fields:
    name: Required. The resource name of the BrowserDlpRule using the form: `o
      rganizations/{organization_id}/locations/global/partnerTenants/{partner_
      tenant_id}/browserDlpRules/{browser_dlp_rule_id}`
  """
    name = _messages.StringField(1, required=True)