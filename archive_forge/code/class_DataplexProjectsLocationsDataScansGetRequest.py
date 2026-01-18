from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataScansGetRequest(_messages.Message):
    """A DataplexProjectsLocationsDataScansGetRequest object.

  Enums:
    ViewValueValuesEnum: Optional. Select the DataScan view to return.
      Defaults to BASIC.

  Fields:
    name: Required. The resource name of the dataScan:
      projects/{project}/locations/{location_id}/dataScans/{data_scan_id}
      where project refers to a project_id or project_number and location_id
      refers to a GCP region.
    view: Optional. Select the DataScan view to return. Defaults to BASIC.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. Select the DataScan view to return. Defaults to BASIC.

    Values:
      DATA_SCAN_VIEW_UNSPECIFIED: The API will default to the BASIC view.
      BASIC: Basic view that does not include spec and result.
      FULL: Include everything.
    """
        DATA_SCAN_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)