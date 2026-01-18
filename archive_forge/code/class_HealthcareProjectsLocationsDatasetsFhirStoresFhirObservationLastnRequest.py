from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirObservationLastnRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsFhirStoresFhirObservationLastnRequest
  object.

  Fields:
    parent: Required. Name of the FHIR store to retrieve resources from.
  """
    parent = _messages.StringField(1, required=True)