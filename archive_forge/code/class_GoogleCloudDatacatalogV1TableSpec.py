from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1TableSpec(_messages.Message):
    """Normal BigQuery table specification.

  Fields:
    groupedEntry: Output only. If the table is date-sharded, that is, it
      matches the `[prefix]YYYYMMDD` name pattern, this field is the Data
      Catalog resource name of the date-sharded grouped entry. For example: `p
      rojects/{PROJECT_ID}/locations/{LOCATION}/entrygroups/{ENTRY_GROUP_ID}/e
      ntries/{ENTRY_ID}`. Otherwise, `grouped_entry` is empty.
  """
    groupedEntry = _messages.StringField(1)