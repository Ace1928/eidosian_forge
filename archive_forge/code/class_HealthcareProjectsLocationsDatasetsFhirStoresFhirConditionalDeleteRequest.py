from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirConditionalDeleteRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsFhirStoresFhirConditionalDeleteRequest
  object.

  Fields:
    parent: The name of the FHIR store this resource belongs to.
    type: The FHIR resource type to delete, such as Patient or Observation.
      For a complete list, see the FHIR Resource Index ([DSTU2](https://hl7.or
      g/implement/standards/fhir/DSTU2/resourcelist.html),
      [STU3](https://hl7.org/implement/standards/fhir/STU3/resourcelist.html),
      [R4](https://hl7.org/implement/standards/fhir/R4/resourcelist.html)).
  """
    parent = _messages.StringField(1, required=True)
    type = _messages.StringField(2, required=True)