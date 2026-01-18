from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigquerydatatransferProjectsDataSourcesGetRequest(_messages.Message):
    """A BigquerydatatransferProjectsDataSourcesGetRequest object.

  Fields:
    name: Required. The field will contain name of the resource requested, for
      example: `projects/{project_id}/dataSources/{data_source_id}` or `projec
      ts/{project_id}/locations/{location_id}/dataSources/{data_source_id}`
  """
    name = _messages.StringField(1, required=True)