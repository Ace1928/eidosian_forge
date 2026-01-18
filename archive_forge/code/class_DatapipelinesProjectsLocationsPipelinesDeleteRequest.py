from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatapipelinesProjectsLocationsPipelinesDeleteRequest(_messages.Message):
    """A DatapipelinesProjectsLocationsPipelinesDeleteRequest object.

  Fields:
    name: Required. The pipeline name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID/pipelines/PIPELINE_ID`.
  """
    name = _messages.StringField(1, required=True)