from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesZonesEntitiesCreateRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesZonesEntitiesCreateRequest object.

  Fields:
    googleCloudDataplexV1Entity: A GoogleCloudDataplexV1Entity resource to be
      passed as the request body.
    parent: Required. The resource name of the parent zone: projects/{project_
      number}/locations/{location_id}/lakes/{lake_id}/zones/{zone_id}.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    googleCloudDataplexV1Entity = _messages.MessageField('GoogleCloudDataplexV1Entity', 1)
    parent = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)