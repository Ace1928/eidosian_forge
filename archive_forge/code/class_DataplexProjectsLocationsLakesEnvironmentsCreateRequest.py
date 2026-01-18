from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesEnvironmentsCreateRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesEnvironmentsCreateRequest object.

  Fields:
    environmentId: Required. Environment identifier. * Must contain only
      lowercase letters, numbers and hyphens. * Must start with a letter. *
      Must be between 1-63 characters. * Must end with a number or a letter. *
      Must be unique within the lake.
    googleCloudDataplexV1Environment: A GoogleCloudDataplexV1Environment
      resource to be passed as the request body.
    parent: Required. The resource name of the parent lake:
      projects/{project_id}/locations/{location_id}/lakes/{lake_id}.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    environmentId = _messages.StringField(1)
    googleCloudDataplexV1Environment = _messages.MessageField('GoogleCloudDataplexV1Environment', 2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)