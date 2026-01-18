from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DataSourceType(_messages.Message):
    """Message used to identify the type of resource being profiled.

  Fields:
    dataSource: Output only. An identifying string to the type of resource
      being profiled. Current values: google/bigquery/table, google/project
  """
    dataSource = _messages.StringField(1)