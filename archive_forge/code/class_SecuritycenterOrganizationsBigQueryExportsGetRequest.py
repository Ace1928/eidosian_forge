from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsBigQueryExportsGetRequest(_messages.Message):
    """A SecuritycenterOrganizationsBigQueryExportsGetRequest object.

  Fields:
    name: Required. Name of the BigQuery export to retrieve. Its format is
      organizations/{organization}/bigQueryExports/{export_id},
      folders/{folder}/bigQueryExports/{export_id}, or
      projects/{project}/bigQueryExports/{export_id}
  """
    name = _messages.StringField(1, required=True)