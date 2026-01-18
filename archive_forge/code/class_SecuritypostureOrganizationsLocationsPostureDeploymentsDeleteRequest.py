from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritypostureOrganizationsLocationsPostureDeploymentsDeleteRequest(_messages.Message):
    """A SecuritypostureOrganizationsLocationsPostureDeploymentsDeleteRequest
  object.

  Fields:
    etag: Optional. Etag value of the PostureDeployment to be deleted.
    name: Required. Name of the resource.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)