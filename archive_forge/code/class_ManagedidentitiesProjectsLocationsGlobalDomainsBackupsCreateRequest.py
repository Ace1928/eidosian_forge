from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsBackupsCreateRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalDomainsBackupsCreateRequest
  object.

  Fields:
    backup: A Backup resource to be passed as the request body.
    backupId: Required. Backup Id, unique name to identify the backups with
      the following restrictions: * Must be lowercase letters, numbers, and
      hyphens * Must start with a letter. * Must contain between 1-63
      characters. * Must end with a number or a letter. * Must be unique
      within the domain.
    parent: Required. The domain resource name using the form:
      `projects/{project_id}/locations/global/domains/{domain_name}`
  """
    backup = _messages.MessageField('Backup', 1)
    backupId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)