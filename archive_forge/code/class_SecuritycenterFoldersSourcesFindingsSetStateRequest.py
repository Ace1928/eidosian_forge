from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersSourcesFindingsSetStateRequest(_messages.Message):
    """A SecuritycenterFoldersSourcesFindingsSetStateRequest object.

  Fields:
    name: Required. The [relative resource name](https://cloud.google.com/apis
      /design/resource_names#relative_resource_name) of the finding. If no
      location is specified, finding is assumed to be in global. The following
      list shows some examples: + `organizations/{organization_id}/sources/{so
      urce_id}/findings/{finding_id}` + `organizations/{organization_id}/sourc
      es/{source_id}/locations/{location_id}/findings/{finding_id}` +
      `folders/{folder_id}/sources/{source_id}/findings/{finding_id}` + `folde
      rs/{folder_id}/sources/{source_id}/locations/{location_id}/findings/{fin
      ding_id}` +
      `projects/{project_id}/sources/{source_id}/findings/{finding_id}` + `pro
      jects/{project_id}/sources/{source_id}/locations/{location_id}/findings/
      {finding_id}`
    setFindingStateRequest: A SetFindingStateRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    setFindingStateRequest = _messages.MessageField('SetFindingStateRequest', 2)