from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataScansDeleteRequest(_messages.Message):
    """A DataplexProjectsLocationsDataScansDeleteRequest object.

  Fields:
    name: Required. The resource name of the dataScan:
      projects/{project}/locations/{location_id}/dataScans/{data_scan_id}
      where project refers to a project_id or project_number and location_id
      refers to a GCP region.
  """
    name = _messages.StringField(1, required=True)