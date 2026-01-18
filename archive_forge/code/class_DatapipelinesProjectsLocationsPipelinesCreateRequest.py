from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatapipelinesProjectsLocationsPipelinesCreateRequest(_messages.Message):
    """A DatapipelinesProjectsLocationsPipelinesCreateRequest object.

  Fields:
    googleCloudDatapipelinesV1Pipeline: A GoogleCloudDatapipelinesV1Pipeline
      resource to be passed as the request body.
    parent: Required. The location name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID`.
  """
    googleCloudDatapipelinesV1Pipeline = _messages.MessageField('GoogleCloudDatapipelinesV1Pipeline', 1)
    parent = _messages.StringField(2, required=True)