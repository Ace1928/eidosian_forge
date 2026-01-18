from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersLocationsFindingsBulkMuteRequest(_messages.Message):
    """A SecuritycenterFoldersLocationsFindingsBulkMuteRequest object.

  Fields:
    bulkMuteFindingsRequest: A BulkMuteFindingsRequest resource to be passed
      as the request body.
    parent: Required. The parent, at which bulk action needs to be applied. If
      no location is specified, findings are updated in global. The following
      list shows some examples: + `organizations/[organization_id]` +
      `organizations/[organization_id]/locations/[location_id]` +
      `folders/[folder_id]` + `folders/[folder_id]/locations/[location_id]` +
      `projects/[project_id]` +
      `projects/[project_id]/locations/[location_id]`
  """
    bulkMuteFindingsRequest = _messages.MessageField('BulkMuteFindingsRequest', 1)
    parent = _messages.StringField(2, required=True)