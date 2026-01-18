from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsCreateRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsCreateRequest object.

  Fields:
    dataset: A Dataset resource to be passed as the request body.
    datasetId: Required. The ID of the dataset that is being created. The
      string must match the following regex: `[\\p{L}\\p{N}_\\-\\.]{1,256}`.
    parent: Required. The name of the project in which the server creates the
      dataset. For example,`projects/{project_id}/locations/{location_id}`.
  """
    dataset = _messages.MessageField('Dataset', 1)
    datasetId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)