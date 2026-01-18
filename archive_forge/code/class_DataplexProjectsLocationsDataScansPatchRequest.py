from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataScansPatchRequest(_messages.Message):
    """A DataplexProjectsLocationsDataScansPatchRequest object.

  Fields:
    googleCloudDataplexV1DataScan: A GoogleCloudDataplexV1DataScan resource to
      be passed as the request body.
    name: Output only. The relative resource name of the scan, of the form:
      projects/{project}/locations/{location_id}/dataScans/{datascan_id},
      where project refers to a project_id or project_number and location_id
      refers to a GCP region.
    updateMask: Required. Mask of fields to update.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    googleCloudDataplexV1DataScan = _messages.MessageField('GoogleCloudDataplexV1DataScan', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)