from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsCheckMigrationPermissionRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalDomainsCheckMigrationPermissio
  nRequest object.

  Fields:
    checkMigrationPermissionRequest: A CheckMigrationPermissionRequest
      resource to be passed as the request body.
    domain: Required. The domain resource name using the form:
      `projects/{project_id}/locations/global/domains/{domain_name}`
  """
    checkMigrationPermissionRequest = _messages.MessageField('CheckMigrationPermissionRequest', 1)
    domain = _messages.StringField(2, required=True)