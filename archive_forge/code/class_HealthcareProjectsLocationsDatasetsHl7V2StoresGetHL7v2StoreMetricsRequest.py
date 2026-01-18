from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsHl7V2StoresGetHL7v2StoreMetricsRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsHl7V2StoresGetHL7v2StoreMetricsRequest
  object.

  Fields:
    name: Required. The resource name of the HL7v2 store to get metrics for,
      in the format `projects/{project_id}/locations/{location_id}/datasets/{d
      ataset_id}/hl7V2Stores/{hl7v2_store_id}`.
  """
    name = _messages.StringField(1, required=True)