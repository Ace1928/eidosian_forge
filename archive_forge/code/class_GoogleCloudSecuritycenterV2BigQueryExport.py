from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2BigQueryExport(_messages.Message):
    """Configures how to deliver Findings to BigQuery Instance.

  Fields:
    createTime: Output only. The time at which the BigQuery export was
      created. This field is set by the server and will be ignored if provided
      on export on creation.
    dataset: The dataset to write findings' updates to. Its format is
      "projects/[project_id]/datasets/[bigquery_dataset_id]". BigQuery Dataset
      unique ID must contain only letters (a-z, A-Z), numbers (0-9), or
      underscores (_).
    description: The description of the export (max of 1024 characters).
    filter: Expression that defines the filter to apply across create/update
      events of findings. The expression is a list of zero or more
      restrictions combined via logical operators `AND` and `OR`. Parentheses
      are supported, and `OR` has higher precedence than `AND`. Restrictions
      have the form ` ` and may have a `-` character in front of them to
      indicate negation. The fields map to those defined in the corresponding
      resource. The supported operators are: * `=` for all value types. * `>`,
      `<`, `>=`, `<=` for integer values. * `:`, meaning substring matching,
      for strings. The supported value types are: * string literals in quotes.
      * integer literals without quotes. * boolean literals `true` and `false`
      without quotes.
    mostRecentEditor: Output only. Email address of the user who last edited
      the BigQuery export. This field is set by the server and will be ignored
      if provided on export creation or update.
    name: The relative resource name of this export. See: https://cloud.google
      .com/apis/design/resource_names#relative_resource_name. The following
      list shows some examples: + `organizations/{organization_id}/locations/{
      location_id}/bigQueryExports/{export_id}` + `folders/{folder_id}/locatio
      ns/{location_id}/bigQueryExports/{export_id}` + `projects/{project_id}/l
      ocations/{location_id}/bigQueryExports/{export_id}` This field is
      provided in responses, and is ignored when provided in create requests.
    principal: Output only. The service account that needs permission to
      create table and upload data to the BigQuery dataset.
    updateTime: Output only. The most recent time at which the BigQuery export
      was updated. This field is set by the server and will be ignored if
      provided on export creation or update.
  """
    createTime = _messages.StringField(1)
    dataset = _messages.StringField(2)
    description = _messages.StringField(3)
    filter = _messages.StringField(4)
    mostRecentEditor = _messages.StringField(5)
    name = _messages.StringField(6)
    principal = _messages.StringField(7)
    updateTime = _messages.StringField(8)