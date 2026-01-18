from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesEnvironmentsPatchRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesEnvironmentsPatchRequest object.

  Fields:
    googleCloudDataplexV1Environment: A GoogleCloudDataplexV1Environment
      resource to be passed as the request body.
    name: Output only. The relative resource name of the environment, of the
      form: projects/{project_id}/locations/{location_id}/lakes/{lake_id}/envi
      ronment/{environment_id}
    updateMask: Required. Mask of fields to update.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    googleCloudDataplexV1Environment = _messages.MessageField('GoogleCloudDataplexV1Environment', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)