from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterProjectsBigQueryExportsPatchRequest(_messages.Message):
    """A SecuritycenterProjectsBigQueryExportsPatchRequest object.

  Fields:
    googleCloudSecuritycenterV1BigQueryExport: A
      GoogleCloudSecuritycenterV1BigQueryExport resource to be passed as the
      request body.
    name: The relative resource name of this export. See: https://cloud.google
      .com/apis/design/resource_names#relative_resource_name. Example format:
      "organizations/{organization_id}/bigQueryExports/{export_id}" Example
      format: "folders/{folder_id}/bigQueryExports/{export_id}" Example
      format: "projects/{project_id}/bigQueryExports/{export_id}" This field
      is provided in responses, and is ignored when provided in create
      requests.
    updateMask: The list of fields to be updated. If empty all mutable fields
      will be updated.
  """
    googleCloudSecuritycenterV1BigQueryExport = _messages.MessageField('GoogleCloudSecuritycenterV1BigQueryExport', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)