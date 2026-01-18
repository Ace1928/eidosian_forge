from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnrollDataSourcesRequest(_messages.Message):
    """A request to enroll a set of data sources so they are visible in the
  BigQuery UI's `Transfer` tab.

  Fields:
    dataSourceIds: Data sources that are enrolled. It is required to provide
      at least one data source id.
  """
    dataSourceIds = _messages.StringField(1, repeated=True)