from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesZonesPatchRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesZonesPatchRequest object.

  Fields:
    googleCloudDataplexV1Zone: A GoogleCloudDataplexV1Zone resource to be
      passed as the request body.
    name: Output only. The relative resource name of the zone, of the form: pr
      ojects/{project_number}/locations/{location_id}/lakes/{lake_id}/zones/{z
      one_id}.
    updateMask: Required. Mask of fields to update.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    googleCloudDataplexV1Zone = _messages.MessageField('GoogleCloudDataplexV1Zone', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)