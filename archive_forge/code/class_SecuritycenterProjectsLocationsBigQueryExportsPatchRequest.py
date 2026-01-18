from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterProjectsLocationsBigQueryExportsPatchRequest(_messages.Message):
    """A SecuritycenterProjectsLocationsBigQueryExportsPatchRequest object.

  Fields:
    googleCloudSecuritycenterV2BigQueryExport: A
      GoogleCloudSecuritycenterV2BigQueryExport resource to be passed as the
      request body.
    name: The relative resource name of this export. See: https://cloud.google
      .com/apis/design/resource_names#relative_resource_name. The following
      list shows some examples: + `organizations/{organization_id}/locations/{
      location_id}/bigQueryExports/{export_id}` + `folders/{folder_id}/locatio
      ns/{location_id}/bigQueryExports/{export_id}` + `projects/{project_id}/l
      ocations/{location_id}/bigQueryExports/{export_id}` This field is
      provided in responses, and is ignored when provided in create requests.
    updateMask: The list of fields to be updated. If empty all mutable fields
      will be updated.
  """
    googleCloudSecuritycenterV2BigQueryExport = _messages.MessageField('GoogleCloudSecuritycenterV2BigQueryExport', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)