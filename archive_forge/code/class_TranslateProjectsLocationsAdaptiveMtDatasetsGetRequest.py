from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsAdaptiveMtDatasetsGetRequest(_messages.Message):
    """A TranslateProjectsLocationsAdaptiveMtDatasetsGetRequest object.

  Fields:
    name: Required. Name of the dataset. In the form of `projects/{project-
      number-or-id}/locations/{location-id}/adaptiveMtDatasets/{adaptive-mt-
      dataset-id}`
  """
    name = _messages.StringField(1, required=True)