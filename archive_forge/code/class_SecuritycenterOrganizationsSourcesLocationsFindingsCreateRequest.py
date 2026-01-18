from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsSourcesLocationsFindingsCreateRequest(_messages.Message):
    """A SecuritycenterOrganizationsSourcesLocationsFindingsCreateRequest
  object.

  Fields:
    findingId: Required. Unique identifier provided by the client within the
      parent scope. It must be alphanumeric and less than or equal to 32
      characters and greater than 0 characters in length.
    googleCloudSecuritycenterV2Finding: A GoogleCloudSecuritycenterV2Finding
      resource to be passed as the request body.
    parent: Required. Resource name of the new finding's parent. The following
      list shows some examples of the format: +
      `organizations/[organization_id]/sources/[source_id]` + `organizations/[
      organization_id]/sources/[source_id]/locations/[location_id]`
  """
    findingId = _messages.StringField(1)
    googleCloudSecuritycenterV2Finding = _messages.MessageField('GoogleCloudSecuritycenterV2Finding', 2)
    parent = _messages.StringField(3, required=True)