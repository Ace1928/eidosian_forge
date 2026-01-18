from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterProjectsSourcesFindingsPatchRequest(_messages.Message):
    """A SecuritycenterProjectsSourcesFindingsPatchRequest object.

  Fields:
    googleCloudSecuritycenterV2Finding: A GoogleCloudSecuritycenterV2Finding
      resource to be passed as the request body.
    name: The [relative resource name](https://cloud.google.com/apis/design/re
      source_names#relative_resource_name) of the finding. The following list
      shows some examples: + `organizations/{organization_id}/sources/{source_
      id}/findings/{finding_id}` + `organizations/{organization_id}/sources/{s
      ource_id}/locations/{location_id}/findings/{finding_id}` +
      `folders/{folder_id}/sources/{source_id}/findings/{finding_id}` + `folde
      rs/{folder_id}/sources/{source_id}/locations/{location_id}/findings/{fin
      ding_id}` +
      `projects/{project_id}/sources/{source_id}/findings/{finding_id}` + `pro
      jects/{project_id}/sources/{source_id}/locations/{location_id}/findings/
      {finding_id}`
    updateMask: The FieldMask to use when updating the finding resource. This
      field should not be specified when creating a finding. When updating a
      finding, an empty mask is treated as updating all mutable fields and
      replacing source_properties. Individual source_properties can be
      added/updated by using "source_properties." in the field mask.
  """
    googleCloudSecuritycenterV2Finding = _messages.MessageField('GoogleCloudSecuritycenterV2Finding', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)