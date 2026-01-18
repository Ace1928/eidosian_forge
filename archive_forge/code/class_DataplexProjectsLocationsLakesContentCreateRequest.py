from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesContentCreateRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesContentCreateRequest object.

  Fields:
    googleCloudDataplexV1Content: A GoogleCloudDataplexV1Content resource to
      be passed as the request body.
    parent: Required. The resource name of the parent lake:
      projects/{project_id}/locations/{location_id}/lakes/{lake_id}
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    googleCloudDataplexV1Content = _messages.MessageField('GoogleCloudDataplexV1Content', 1)
    parent = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)