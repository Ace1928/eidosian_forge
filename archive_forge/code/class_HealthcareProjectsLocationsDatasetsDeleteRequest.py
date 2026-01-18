from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDeleteRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsDeleteRequest object.

  Fields:
    name: Required. The name of the dataset to delete. For example,
      `projects/{project_id}/locations/{location_id}/datasets/{dataset_id}`.
  """
    name = _messages.StringField(1, required=True)