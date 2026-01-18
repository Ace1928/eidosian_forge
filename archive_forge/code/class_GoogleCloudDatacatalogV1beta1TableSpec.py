from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1TableSpec(_messages.Message):
    """Normal BigQuery table spec.

  Fields:
    groupedEntry: Output only. If the table is a dated shard, i.e., with name
      pattern `[prefix]YYYYMMDD`, `grouped_entry` is the Data Catalog resource
      name of the date sharded grouped entry, for example, `projects/{project_
      id}/locations/{location}/entrygroups/{entry_group_id}/entries/{entry_id}
      `. Otherwise, `grouped_entry` is empty.
  """
    groupedEntry = _messages.StringField(1)