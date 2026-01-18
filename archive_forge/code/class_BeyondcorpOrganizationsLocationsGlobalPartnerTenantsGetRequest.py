from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpOrganizationsLocationsGlobalPartnerTenantsGetRequest(_messages.Message):
    """A BeyondcorpOrganizationsLocationsGlobalPartnerTenantsGetRequest object.

  Fields:
    name: Required. The resource name of the PartnerTenant using the form: `or
      ganizations/{organization_id}/locations/global/partnerTenants/{partner_t
      enant_id}`
  """
    name = _messages.StringField(1, required=True)