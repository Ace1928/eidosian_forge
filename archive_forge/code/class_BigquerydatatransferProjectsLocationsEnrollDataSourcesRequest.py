from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigquerydatatransferProjectsLocationsEnrollDataSourcesRequest(_messages.Message):
    """A BigquerydatatransferProjectsLocationsEnrollDataSourcesRequest object.

  Fields:
    enrollDataSourcesRequest: A EnrollDataSourcesRequest resource to be passed
      as the request body.
    name: Required. The name of the project resource in the form:
      `projects/{project_id}`
  """
    enrollDataSourcesRequest = _messages.MessageField('EnrollDataSourcesRequest', 1)
    name = _messages.StringField(2, required=True)