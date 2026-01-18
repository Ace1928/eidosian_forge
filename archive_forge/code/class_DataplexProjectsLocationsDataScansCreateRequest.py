from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataScansCreateRequest(_messages.Message):
    """A DataplexProjectsLocationsDataScansCreateRequest object.

  Fields:
    dataScanId: Required. DataScan identifier. Must contain only lowercase
      letters, numbers and hyphens. Must start with a letter. Must end with a
      number or a letter. Must be between 1-63 characters. Must be unique
      within the customer project / location.
    googleCloudDataplexV1DataScan: A GoogleCloudDataplexV1DataScan resource to
      be passed as the request body.
    parent: Required. The resource name of the parent location:
      projects/{project}/locations/{location_id} where project refers to a
      project_id or project_number and location_id refers to a GCP region.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    dataScanId = _messages.StringField(1)
    googleCloudDataplexV1DataScan = _messages.MessageField('GoogleCloudDataplexV1DataScan', 2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)