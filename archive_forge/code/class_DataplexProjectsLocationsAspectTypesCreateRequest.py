from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsAspectTypesCreateRequest(_messages.Message):
    """A DataplexProjectsLocationsAspectTypesCreateRequest object.

  Fields:
    aspectTypeId: Required. AspectType identifier.
    googleCloudDataplexV1AspectType: A GoogleCloudDataplexV1AspectType
      resource to be passed as the request body.
    parent: Required. The resource name of the AspectType, of the form:
      projects/{project_number}/locations/{location_id} where location_id
      refers to a GCP region.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    aspectTypeId = _messages.StringField(1)
    googleCloudDataplexV1AspectType = _messages.MessageField('GoogleCloudDataplexV1AspectType', 2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)