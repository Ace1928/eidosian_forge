from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersLocationsBigQueryExportsGetRequest(_messages.Message):
    """A SecuritycenterFoldersLocationsBigQueryExportsGetRequest object.

  Fields:
    name: Required. Name of the BigQuery export to retrieve. The following
      list shows some examples of the format: + `organizations/{organization}/
      locations/{location}/bigQueryExports/{export_id}` +
      `folders/{folder}/locations/{location}/bigQueryExports/{export_id}` +
      `projects/{project}locations/{location}//bigQueryExports/{export_id}`
  """
    name = _messages.StringField(1, required=True)