from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersLocationsBigQueryExportsDeleteRequest(_messages.Message):
    """A SecuritycenterFoldersLocationsBigQueryExportsDeleteRequest object.

  Fields:
    name: Required. The name of the BigQuery export to delete. The following
      list shows some examples of the format: + `organizations/{organization}/
      locations/{location}/bigQueryExports/{export_id}` +
      `folders/{folder}/locations/{location}/bigQueryExports/{export_id}` +
      `projects/{project}/locations/{location}/bigQueryExports/{export_id}`
  """
    name = _messages.StringField(1, required=True)